

def train_model_sphere(model, d=3, h=0.1, lam=1.0, T=2000, batch_size=1000, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Using device:", device)
    print("Iteration, Loss")

    for i in range(T + 1):
        
        x = sample_unit_ball(batch_size, d, device)

        # Finite Difference Approximation
        grad_u_fd = torch.zeros_like(x)
        for j in range(d):
            ei = torch.zeros_like(x)
            ei[:, j] = h

            u_plus = model(x + ei)
            u_minus = model(x - ei)

            # Centered difference
            #grad_j = (u_plus - u_minus) / (2 * h)
            # Monotone scheme
            grad_j = torch.max(m(model(x)-u_minus), m(model(x)-u_plus))/h
            grad_u_fd[:, j] = grad_j.squeeze()

        grad_norm_fd = torch.norm(grad_u_fd, dim=1, keepdim=True)
        loss = lam * torch.mean((grad_norm_fd - 1)**2)

        # Boundary condition
        boundary_x = torch.randn_like(x)
        boundary_x = boundary_x / torch.norm(boundary_x, dim=1, keepdim=True)
        # u_b≈0
        u_b = model(boundary_x)
        loss += torch.mean(u_b**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"{i},{loss.item():.6f}")

    return model



def train_model_cube(model, d=3, h=0.1, lam=1.0, T=2000, batch_size=1000, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Using device:", device)
    print("Iteration, Loss")

    for i in range(T + 1):
        
        # Sample uniformly in [-1, 1]^3
        x = (2 * torch.rand(batch_size, d) - 1).to(device)
        
        grad_u_fd = torch.zeros_like(x)
        for j in range(d):
            ei = torch.zeros_like(x)
            ei[:, j] = h

            u_plus = model(x + ei)
            u_minus = model(x - ei)

            # Centered difference
            #grad_j = (u_plus - u_minus) / (2 * h)
            # Monotone scheme
            grad_j = torch.max(m(model(x)-u_minus), m(model(x)-u_plus))/h
            grad_u_fd[:, j] = grad_j.squeeze()

        grad_norm_fd = torch.norm(grad_u_fd, dim=1, keepdim=True)
        loss = lam * torch.mean((grad_norm_fd - 1)**2)

        # Boundary condition on the cube [-1, 1]^3
        # The cube has 6 faces and we want to divide the batch size equally among the 6 faces
        
        num_faces = 2 * d  # 6 faces 
        boundary_points_per_face = batch_size // num_faces # split sizes for each face 
        boundary_x_list = []
        
       
        for j in range(d):         # j = 0, 1, 2 referes to (x, y, z)
            for side in [-1.0, 1.0]:
                x_face = 2 * torch.rand(boundary_points_per_face, d) - 1
                x_face[:, j] = side  # Fix the j-th coordinate to ±1
                boundary_x_list.append(x_face)

        boundary_x = torch.cat(boundary_x_list, dim=0).to(device)
        u_b = model(boundary_x)
        loss += torch.mean(u_b**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"{i},{loss.item():.6f}")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_network()
model = train_model_sphere(model, device=device)
plot_surface(model, grid_size=100,title='Monotone scheme',color='viridis')

