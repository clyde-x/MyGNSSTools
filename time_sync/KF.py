import numpy as np


class RelativeClockKF:
    """
    Kalman filter: clock + per-frequency ambiguity estimation.

    State: x = [clock, N_L1^s1, N_L2^s1, N_L1^s2, N_L2^s2, ...]

    Observations (per satellite):
        P* (P1, P2, PC)  = clock
        L1               = clock + N_L1
        L2               = clock + N_L2

    H_P = [1, 0, ..., 0, 0, ...]
    H_L1 = [1, ..., 1, 0, ...]
    H_L2 = [1, ..., 0, 1, ...]
    """

    L_KEYS = ["L1", "L2", "LC"]
    P_KEYS = ["P1", "P2", "PC", "C1", "C2"]

    def __init__(
        self,
        tau=1e6,
        sigma_clock=0.3,
        sigma_N=1e-8,
        sigma_N_init=1.0,
        sigma_P=2.0,
        sigma_L=0.02,
    ):
        self.tau = tau
        self.sigma_clock = sigma_clock
        self.sigma_N = sigma_N
        self.q_N = sigma_N ** 2
        self.sigma_N_init = sigma_N_init
        self.sigma_P = sigma_P
        self.sigma_L = sigma_L

        # sat_map: {sat_id: (idx_N_L1, idx_N_L2, idx_N_LC)}
        self.sat_map = {}
        self._last_seen = {}
        self.x = np.zeros(1)          # clock only at start
        self.P = np.eye(1) * 1e4

        self.clock_ref = None

        # ----- diagnostic logging -----
        self._log_enabled = False
        self._log_verbose = False
        self._log_entries = []
        self._pred_info = {}

    # ================================================================
    #  logging API
    # ================================================================
    def enable_logging(self, verbose=False):
        self._log_enabled = True
        self._log_verbose = verbose
        self._log_entries = []
        self._pred_info = {}

    def get_log(self):
        return self._log_entries

    # ================================================================
    #  observation key helpers
    # ================================================================
    @staticmethod
    def _get_p_vals(obs):
        """Return dict {key: value} for all P* / C* entries in obs."""
        return {k: obs[k] for k in RelativeClockKF.P_KEYS if k in obs}

    @staticmethod
    def _get_l_vals(obs):
        """Return dict {key: value} for all L* entries in obs."""
        return {k: obs[k] for k in RelativeClockKF.L_KEYS if k in obs}

    @staticmethod
    def _get_first_p(obs):
        """Return first available P-type observation value, or None."""
        for k in RelativeClockKF.P_KEYS:
            if k in obs:
                return obs[k]
        return None

    # ================================================================
    #  state management
    # ================================================================
    def add_sat(self, sat, n_init_L1=0.0, n_init_L2=0.0, n_init_LC=0.0):
        """Add satellite with per-frequency ambiguity states.

        Each satellite reserves 3 slots: one for each L-key (L1, L2, LC).
        Slots are only activated if the corresponding L observation appears.
        """
        if sat in self.sat_map:
            return
        base_idx = len(self.x)
        # L1, L2, LC — three N slots per satellite
        n_new = 3
        self.sat_map[sat] = (base_idx, base_idx + 1, base_idx + 2)
        self.x = np.concatenate([self.x, [n_init_L1, n_init_L2, n_init_LC]])
        n_old = len(self.P)
        p_new = np.zeros((n_old + n_new, n_old + n_new))
        p_new[:n_old, :n_old] = self.P
        for j in range(n_new):
            p_new[n_old + j, n_old + j] = self.sigma_N_init ** 2
        self.P = p_new
        self._last_seen[sat] = 0

    def _prune_stale(self, max_age=120):
        to_remove = [s for s, age in list(self._last_seen.items()) if age > max_age]
        for sat in to_remove:
            if sat not in self.sat_map:
                del self._last_seen[sat]
                continue
            idx_L1, idx_L2, idx_LC = self.sat_map[sat]
            keep = np.ones(len(self.x), dtype=bool)
            keep[idx_L1] = False
            keep[idx_L2] = False
            keep[idx_LC] = False
            self.x = self.x[keep]
            self.P = self.P[keep][:, keep]
            del self.sat_map[sat]
            del self._last_seen[sat]
            # remap remaining satellites
            removed = sorted([idx_L1, idx_L2, idx_LC], reverse=True)
            new_map = {}
            for s, (i1, i2, i3) in self.sat_map.items():
                for r in removed:
                    i1 = i1 - 1 if i1 > r else i1
                    i2 = i2 - 1 if i2 > r else i2
                    i3 = i3 - 1 if i3 > r else i3
                new_map[s] = (i1, i2, i3)
            self.sat_map = new_map

    def remove_sat(self, sat):
        if sat not in self.sat_map:
            return
        self._last_seen[sat] = 999
        self._prune_stale(max_age=0)

    def reset_sat(self, sat):
        if sat not in self.sat_map:
            return
        for idx in self.sat_map[sat]:
            self.x[idx] = 0.0
            self.P[idx, idx] = self.sigma_N_init ** 2

    def predict(self, dt):
        if dt <= 0:
            return
        phi = np.exp(-dt / self.tau)
        n = len(self.x)
        F = np.eye(n)
        F[0, 0] = phi
        q_clock = self.sigma_clock ** 2 * (1 - phi ** 2)
        Q = np.zeros((n, n))
        Q[0, 0] = q_clock
        if n > 1:
            Q[1:, 1:] = self.q_N * np.eye(n - 1)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        if self._log_enabled:
            self._pred_info = {
                'dt': dt,
                'x': self.x.copy(),
                'P_diag': np.diag(self.P).copy(),
            }

    def update(self, observations, elevations=None):
        # ---- determine which L keys are active in this batch ----
        active_L = set()
        for obs in observations:
            active_L.update(self._get_l_vals(obs).keys())

        # ---- initialize clock_ref on first epoch ----
        if self.clock_ref is None:
            p_vals = []
            for obs in observations:
                p_vals.extend(self._get_p_vals(obs).values())
            self.clock_ref = np.median(p_vals) if p_vals else 0.0

        # ---- add new satellites with per-frequency N init ----
        for obs in observations:
            sat = obs["sat"]
            if sat not in self.sat_map:
                l_vals = self._get_l_vals(obs)
                n_L1 = l_vals.get("L1", self._get_first_p(obs) or 0.0) - self.clock_ref if l_vals.get("L1") is not None else 0.0
                n_L2 = l_vals.get("L2", self._get_first_p(obs) or 0.0) - self.clock_ref if l_vals.get("L2") is not None else 0.0
                n_LC = l_vals.get("LC", self._get_first_p(obs) or 0.0) - self.clock_ref if l_vals.get("LC") is not None else 0.0
                self.add_sat(sat, n_init_L1=n_L1, n_init_L2=n_L2, n_init_LC=n_LC)

        # ---- snapshot for logging ----
        if self._log_enabled:
            _x_pre_upd = self.x.copy()

        # ---- pseudorange median (for anchor and L-consistency check) ----
        p_all = []
        for obs in observations:
            p_all.extend(self._get_p_vals(obs).values())
        p_median = np.median(p_all) if p_all else 0.0

        n = len(self.x)
        rows = []
        z_list = []
        r_diag = []
        rp2 = self.sigma_P ** 2
        rl2 = self.sigma_L ** 2
        iclk = 0

        for obs in observations:
            sat = obs["sat"]
            idx_L1, idx_L2, idx_LC = self.sat_map[sat]

            L_KEY_TO_IDX = {"L1": idx_L1, "L2": idx_L2, "LC": idx_LC}

            el = elevations.get(sat, np.radians(45.0)) if elevations else np.radians(45.0)
            w = 1.0 / max(np.sin(el) ** 2, 0.01)

            # ---- P rows: one per P-type key ----
            p_vals = self._get_p_vals(obs)
            for p_val in p_vals.values():
                h_p = np.zeros(n)
                h_p[iclk] = 1.0
                rows.append(h_p)
                z_list.append(p_val)
                r_diag.append(rp2 * w)

            # ---- L rows: one per L-type key ----
            l_vals = self._get_l_vals(obs)
            for l_key, l_val in l_vals.items():
                if l_key not in L_KEY_TO_IDX:
                    continue
                iN = L_KEY_TO_IDX[l_key]
                L_corr = l_val - self.x[iN]
                if abs(L_corr - p_median) < 20.0:
                    h_l = np.zeros(n)
                    h_l[iclk] = 1.0
                    h_l[iN] = 1.0
                    rows.append(h_l)
                    z_list.append(l_val)
                    r_diag.append(rl2 * w)

        # ---- P anchor ----
        if p_all:
            h_anchor = np.zeros(n)
            h_anchor[0] = 1.0
            rows.append(h_anchor)
            z_list.append(p_median)
            r_diag.append(rp2 / len(observations))

        if len(z_list) <= 1:
            return None

        H = np.vstack(rows)
        z_vec = np.array(z_list)
        R = np.diag(r_diag)

        if np.any(np.isnan(self.x)) or np.any(np.isnan(self.P)):
            return None

        x_save = self.x.copy()
        P_save = self.P.copy()

        v = z_vec - H @ self.x
        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None

        K = np.clip(K, -1e8, 1e8)
        self.x = self.x + K @ v

        I = np.eye(n)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        if np.any(np.isnan(self.x)) or np.any(np.isnan(self.P)):
            self.x = x_save
            self.P = P_save
            return None

        # ---- track and prune stale satellites ----
        obs_sats = set(o["sat"] for o in observations)
        for sat in list(self._last_seen.keys()):
            if sat in obs_sats:
                self._last_seen[sat] = 0
            else:
                self._last_seen[sat] = self._last_seen.get(sat, 0) + 1
        self._prune_stale(max_age=120)

        # ---- diagnostic recording ----
        if self._log_enabled and self._pred_info:
            pred = self._pred_info
            self._log_entries.append({
                'dt': float(pred['dt']),
                'n_sats': len(observations),
                'n_state': int(n),
                'x_pred_clock': float(_x_pre_upd[0]),
                'x_upd_clock': float(self.x[0]),
                'P_pred_clock': float(pred['P_diag'][0]),
                'P_upd_clock': float(self.P[0, 0]),
            })

        return v

    @property
    def clock(self):
        return self.x[0]

    def ambiguity(self):
        """Return {sat: {'L1': N_L1, 'L2': N_L2, 'LC': N_LC}}."""
        result = {}
        for sat, (i1, i2, i3) in self.sat_map.items():
            result[sat] = {'L1': self.x[i1], 'L2': self.x[i2], 'LC': self.x[i3]}
        return result
