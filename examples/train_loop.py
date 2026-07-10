import torch
import torch_numopt


def train_loop(model, loss_fn, opt, data_loader, epochs=100, max_patience=50):
    objfunc = torch_numopt.SupervisedLearningObjective(model=model, loss_fn=loss_fn, optimizer=opt)

    all_loss = {}
    patience = 0
    for epoch in range(epochs):
        all_loss[epoch + 1] = 0
        for batch_idx, (b_x, b_y) in enumerate(data_loader):
            objfunc.set_data(x=b_x, y=b_y)
            opt.step(objfunc)

            with torch.inference_mode():
                all_loss[epoch + 1] += objfunc.loss(*model.parameters()).item()
        all_loss[epoch + 1] /= len(data_loader)

        print("epoch: ", epoch, end="")
        print(", loss: {}".format(all_loss[epoch + 1]))

        if epoch > 0 and all_loss[epoch] <= all_loss[epoch + 1]:
            patience -= 1
        else:
            patience = max_patience

        if patience <= 0:
            break

    return model, all_loss
