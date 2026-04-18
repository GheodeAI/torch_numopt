import torch_numopt

def train_loop(
    model,
    loss_fn,
    opt,
    data_loader,
    epochs=100,
    max_patience=50
):
    device = next(model.parameters()).device
    all_loss = {}
    patience = 0
    for epoch in range(epochs):
        print("epoch: ", epoch, end="")
        all_loss[epoch + 1] = 0
        for i, (b_x, b_y) in enumerate(data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            pre = model(b_x)
            loss = loss_fn(pre, b_y)
            opt.zero_grad()
            loss.backward()

            # parameter update step based on optimizer
            if isinstance(opt, torch_numopt.CustomOptimizer):
                opt.step(b_x, b_y, loss_fn)
            else:
                opt.step()

            all_loss[epoch + 1] += loss
        all_loss[epoch + 1] /= len(data_loader)
        print(f", loss: {all_loss[epoch + 1].cpu().detach().numpy().item()}")

        if epoch > 0 and all_loss[epoch] <= all_loss[epoch+1]:
            patience -= 1
        else:
            patience = max_patience

        if patience <= 0:
            break
    
    return model, all_loss
