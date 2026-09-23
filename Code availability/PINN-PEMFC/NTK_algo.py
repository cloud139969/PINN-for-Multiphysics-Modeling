import torch


class NTKHandler:
    """
    NTK Weight Adaptive Manager
    Calculate the Jacobian Trace and return the dynamic weights
    """

    def __init__(self, update_every=100, momentum=0.9, subset_size=64):
        self.update_every = update_every
        self.momentum = momentum
        self.subset_size = subset_size
        self.step_counter = 0


        self.current_weights = {}
        self.is_initialized = False

    def _compute_trace(self, residual_tensor, params):
        """

        """
        # 1. Mini-batch Sampling
        N = residual_tensor.shape[0]
        if N > self.subset_size:
            #
            idx = torch.randperm(N)[:self.subset_size]
            res_subset = residual_tensor[idx]
        else:
            res_subset = residual_tensor

        # 2.

        loss_sum = res_subset.sum()

        # 3.
        grads = torch.autograd.grad(loss_sum, params, retain_graph=True, create_graph=False, allow_unused=True)

        # 4.
        trace_val = 0.0
        for g in grads:
            if g is not None:
                trace_val += torch.sum(g ** 2).item()

        return trace_val

    def update_and_get_weights(self, pde_residual, bc_residuals_dict, network_model):
        """

        """
        self.step_counter += 1

        #
        if not self.is_initialized:
            for key in bc_residuals_dict.keys():
                self.current_weights[key] = 1.0
            self.is_initialized = True

        #
        if self.step_counter % self.update_every != 0:
            return self.current_weights

        # --- start NTK ---
        # print(f"[NTK] Updating weights at step {self.step_counter}...")

        params = list(network_model.parameters())

        # 1. base(PDE) Trace

        network_model.zero_grad()
        trace_pde = self._compute_trace(pde_residual, params)
        trace_pde = max(trace_pde, 1e-10)  # 防止除0

        # 2. BC.IC Trace
        for name, res_tensor in bc_residuals_dict.items():
            trace_bc = self._compute_trace(res_tensor, params)
            trace_bc = max(trace_bc, 1e-10)

            # NTK : lambda = Trace_PDE / Trace_BC
            target_weight = trace_pde / trace_bc

            #
            old_w = self.current_weights[name]
            new_w = self.momentum * old_w + (1.0 - self.momentum) * target_weight

            self.current_weights[name] = new_w

        return self.current_weights

