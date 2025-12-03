import numpy as np
import math
from typing import Tuple


class CellularNetworkSimulator:
    """
    Simulator for uplink interference in a 19-cell hexagonal network
    with 3 sectors per cell.

    - 19 hexagonal cells, radius R (relative units).
    - Each cell has 3 sectors (120 degrees each).
    - 1 uplink user per sector (3 users per cell).
    - Path loss: d^{-nu}.
    - Shadow fading: log-normal with std shadow_sigma_dB.
    - Interference: co-directional sectors (same sector index) that
      reuse the same frequency group (depending on reuse factor).

    Reuse modes:
        'reuse1' : universal reuse factor 1 (all cells same frequency)
        'reuse3' : reuse factor 3 across cells (3 cell groups)
        'reuse9' : reuse factor 9 across cells (9 cell groups)

    SIR is computed for the 3 users in the central cell (cell index 0).
    """

    def __init__(
        self,
        cell_radius: float = 1.0,
        pathloss_exp: float = 3.8,
        shadow_sigma_dB: float = 8.0,   # given parameters
        seed: int | None = None,        #add seed for reproducibility, so we can analyze the same results always
    ):
        self.R = cell_radius
        self.nu = pathloss_exp
        self.shadow_sigma_dB = shadow_sigma_dB
        self.rng = np.random.default_rng(seed)

        # Build 19-cell hex layout (axial coordinates + Cartesian centers)
        self.axial_coords, self.cell_centers = self._generate_hex_grid(radius=2)

        self.num_cells = self.cell_centers.shape[0]
        assert self.num_cells == 19
        self.num_sectors = 3

        # Precompute frequency-reuse groups for each cell
        self.cell_groups_3 = self._compute_cell_groups_3()
        self.cell_groups_9 = self._compute_cell_groups_9()

        # Precompute hex polygon (centered at origin) for point-inside tests
        self.hex_polygon = self._regular_hexagon_vertices(self.R)

    def _generate_hex_grid(self, radius: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate axial coordinates (q,r) and corresponding Cartesian centers
        for a hexagonal grid of given radius (here radius=2 -> 19 cells).

        We use a flat-topped hex coordinate system.
        """
        axial_coords = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= radius:
                    axial_coords.append((q, r))

        axial_coords = np.array(axial_coords, dtype=int)

        # Convert axial (q,r) to Cartesian (x,y) for flat-top hexes
        centers = []
        for q, r in axial_coords:
            x = self.R * (3 / 2 * q)
            y = self.R * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
            centers.append((x, y))

        centers = np.array(centers, dtype=float)

        # Put the central cell (0,0) first, just for convenience
        # (otherwise order is arbitrary)
        central_idx = np.where((axial_coords[:, 0] == 0) & (axial_coords[:, 1] == 0))[0][0]
        if central_idx != 0:
            axial_coords[[0, central_idx]] = axial_coords[[central_idx, 0]]
            centers[[0, central_idx]] = centers[[central_idx, 0]]

        return axial_coords, centers

    def _compute_cell_groups_3(self) -> np.ndarray:
        """
        3-coloring of hex grid: assigns each cell to one of 3 groups {0,1,2}.
        A standard scheme is (q - r) mod 3, where (q,r) are axial coords.
        """
        q = self.axial_coords[:, 0]
        r = self.axial_coords[:, 1]
        groups = (q - r) % 3
        return groups

    def _compute_cell_groups_9(self) -> np.ndarray:
        """
        9-coloring of the hex grid for reuse factor 9.
        We use a linear combination of axial coords:

            group = (q + 2*r) mod 9

        This creates 9 groups over the infinite hex grid. In our 19-cell
        radius-2 layout, each group has 1–3 cells, and the central cell
        (q=0,r=0) shares its group with some neighbors, so there IS
        co-channel interference in reuse-9 (but less than in reuse-3).
        """
        q = self.axial_coords[:, 0]
        r = self.axial_coords[:, 1]
        groups = (q + 2 * r) % 9
        return groups


    # HEXAGON / SECTOR GEOMETRY
    @staticmethod
    def _regular_hexagon_vertices(R: float) -> np.ndarray:
        """
        Regular flat-top hexagon of side length R, centered at origin.
        Returns vertices as an array shape (6,2) in counter-clockwise order.
        """
        vertices = []
        # Flat-top orientation, start at angle 0 and step 60 degrees
        for k in range(6):
            angle = math.radians(60 * k)
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            vertices.append((x, y))
        return np.array(vertices, dtype=float)

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        """
        Ray-casting algorithm to check if (x,y) lies inside a polygon.
        polygon: array of shape (N,2)
        """
        inside = False
        n = polygon.shape[0]
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # Check if ray from (x,y) to the right intersects edge (x1,y1)-(x2,y2)
            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            )
            if intersects:
                inside = not inside
        return inside

    @staticmethod
    def _angle_in_sector(theta: float, sector_idx: int) -> bool:
        """
        Check if an angle theta (in radians, range [-pi,pi]) is inside a 120-degree
        sector with fixed global orientation.

        Sector definitions (flat-top hex, arbitrary but consistent):
            sector 0: [-pi/3, +pi/3]
            sector 1: [ +pi/3, +pi ]
            sector 2: [ -pi,   -pi/3]
        """
        if sector_idx == 0:
            return -math.pi / 3 <= theta <= math.pi / 3
        elif sector_idx == 1:
            return math.pi / 3 < theta <= math.pi
        elif sector_idx == 2:
            return -math.pi <= theta < -math.pi / 3
        else:
            raise ValueError("sector_idx must be 0, 1, or 2")

    def _sample_point_in_sector(self, cell_center: np.ndarray, sector_idx: int) -> np.ndarray:
        """
        Rejection sampling: generate a point uniformly at random inside the
        intersection of:
            - the hexagon of radius R around cell_center, and
            - the 120-degree sector 'sector_idx'.

        We sample uniformly from the bounding box of the hexagon and reject
        points that are outside the hex or not in the sector.
        """
        cx, cy = cell_center
        R = self.R
        # Bounding box for flat-top hex: x in [-R,R], y in [-sqrt(3)/2*R, +sqrt(3)/2*R]
        y_max_abs = math.sqrt(3) / 2 * R

        while True:
            x = self.rng.uniform(-R, R)
            y = self.rng.uniform(-y_max_abs, y_max_abs)

            # Check inside hex (centered at origin)
            if not self._point_in_polygon(x, y, self.hex_polygon):
                continue

            # Check inside correct angular sector
            theta = math.atan2(y, x)  # angle in [-pi,pi]
            if not self._angle_in_sector(theta, sector_idx):
                continue

            # Translate to actual cell center
            return np.array([cx + x, cy + y], dtype=float)

    # SHADOW FADING
    def _shadow_fading_linear(self, size: int | Tuple[int, ...] = 1) -> np.ndarray:
        """
        Log-normal shadow fading in linear scale.

        X_dB ~ N(0, sigma^2),  SF_linear = 10^{X_dB / 10}.
        """
        X_dB = self.rng.normal(loc=0.0, scale=self.shadow_sigma_dB, size=size)
        return np.power(10.0, X_dB / 10.0)

    # USER GENERATION
    def generate_user_positions(self) -> np.ndarray:
        """
        Generate user positions for one Monte-Carlo snapshot.

        Returns
        -------
        positions : ndarray, shape (num_cells, num_sectors, 2)
            positions[c, s, :] = (x,y) of user in cell c, sector s.
        """
        positions = np.zeros((self.num_cells, self.num_sectors, 2), dtype=float)
        for c in range(self.num_cells):
            center = self.cell_centers[c]
            for s in range(self.num_sectors):
                positions[c, s, :] = self._sample_point_in_sector(center, s)
        return positions

    # FREQUENCY REUSE LOGIC
    def _cells_share_frequency(self, c1: int, c2: int, reuse_mode: str) -> bool:
        """
        Return True if cells c1 and c2 reuse the same frequency for a given
        reuse configuration.
        """
        if reuse_mode == "reuse1":
            return True
        elif reuse_mode == "reuse3":
            return self.cell_groups_3[c1] == self.cell_groups_3[c2]
        elif reuse_mode == "reuse9":
            return self.cell_groups_9[c1] == self.cell_groups_9[c2]
        else:
            raise ValueError("Unknown reuse_mode. Use 'reuse1', 'reuse3', or 'reuse9'.")

    # SIR COMPUTATION
    def compute_sir_snapshot(self, reuse_mode: str = "reuse1") -> np.ndarray:
        """
        Compute SIR (linear) for the 3 users in the central cell (cell 0),
        for a single snapshot.

        Steps:
            - Generate user positions in all cells/sectors.
            - For each serving sector s in {0,1,2} of cell 0:
                * Compute desired signal: PL_serv * SF_serv.
                * Sum interference from all *other* cells that:
                      · share frequency (according to reuse_mode), and
                      · are co-directional (same sector index s).

        Parameters
        ----------
        reuse_mode : {'reuse1','reuse3','reuse9'}

        Returns
        -------
        sir_linear : ndarray, shape (3,)
            SIR values (linear scale) for sectors 0,1,2 of the central cell.
        """
        positions = self.generate_user_positions()
        sir = np.zeros(3, dtype=float)

        center0 = self.cell_centers[0]

        for s in range(self.num_sectors):
            # Serving link
            user_serv = positions[0, s, :]
            d_serv = np.linalg.norm(user_serv - center0)
            # Guard for degenerate very small distance
            d_serv = max(d_serv, 1e-6)

            PL_serv = d_serv ** (-self.nu)
            SF_serv = self._shadow_fading_linear()[0]
            desired_power = PL_serv * SF_serv

            # Interference
            interference = 0.0
            for c in range(1, self.num_cells):  # exclude central cell
                if not self._cells_share_frequency(c, 0, reuse_mode):
                    continue  # different frequency group → no interference

                # Co-directional: same sector index s
                user_int = positions[c, s, :]
                d_int = np.linalg.norm(user_int - center0)
                d_int = max(d_int, 1e-6)

                PL_int = d_int ** (-self.nu)
                SF_int = self._shadow_fading_linear()[0]
                interference += PL_int * SF_int

            if interference <= 0.0:
                sir[s] = np.inf  # no interferers in this configuration
            else:
                sir[s] = desired_power / interference

        return sir

    # MONTE CARLO DRIVER
    def run_monte_carlo(
        self, num_snapshots: int, reuse_mode: str = "reuse1"
    ) -> np.ndarray:
        """
        Run many snapshots and collect SIR samples (linear scale)
        for the 3 sectors of the central cell.

        Parameters
        ----------
        num_snapshots : int
            Number of independent Monte-Carlo snapshots.
        reuse_mode : {'reuse1','reuse3','reuse9'}

        Returns
        -------
        sir_samples : ndarray, shape (num_snapshots, 3)
            sir_samples[i, s] is SIR (linear) for snapshot i, sector s.
        """
        sir_samples = np.zeros((num_snapshots, self.num_sectors), dtype=float)
        for i in range(num_snapshots):
            sir_samples[i, :] = self.compute_sir_snapshot(reuse_mode=reuse_mode)
        return sir_samples

    # CONVENIENCE: CONVERT TO dB
    @staticmethod
    def linear_to_dB(x: np.ndarray) -> np.ndarray:
        """
        Convert linear values to dB. Handles np.inf gracefully.
        """
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        finite_mask = np.isfinite(x) & (x > 0)
        out[finite_mask] = 10 * np.log10(x[finite_mask])
        out[~finite_mask] = 100.0  # arbitrary large number for +inf
        return out



