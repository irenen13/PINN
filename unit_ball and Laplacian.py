# === Sampling a unit ball
def sample_unit_ball(n, d, device):
    x = torch.randn(n, d, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True)
    r = torch.rand(n, 1, device=device) ** (1.0 / d) # make it uniform
    return r * x


# === Compute Laplacian with exact derivative===
def compute_laplacian(u, x):
    laplacian = 0
    for j in range(x.size(1)):
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        second_deriv = torch.autograd.grad(grad_u[:, j].sum(), x, create_graph=True)[0][:, j]
        laplacian += second_deriv
    return laplacian.view(-1, 1)

