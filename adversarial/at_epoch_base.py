import tqdm


def adv_train(model, train_dataloader, optimizer, loss_fun, attacks: list, epoch=1, device="cuda"):
    model.train()
    for i in range(epoch):
        for attack in attacks:
            for (x, y) in tqdm.tqdm(train_dataloader, desc=f"[INFO] AT Epoch {i + 1}"):
                optimizer.zero_grad()
                if attack:
                    x = attack(model, x.to(device)).clone().detach()
                predict = model(x.to(device))
                losses = loss_fun(predict, y.to(device))
                losses.backward()
                optimizer.step()