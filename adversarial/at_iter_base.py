import tqdm


def iter_step(model, x, y, optimizer, loss_fun, attack=None, device="cuda"):
    if attack:
        x = attack(model, x.to(device)).clone().detach()
    optimizer.zero_grad()
    predict = model(x.to(device))
    losses = loss_fun(predict, y.to(device))
    losses.backward()
    optimizer.step()


def adv_train(model, train_dataloader, loss_fun, optimizer, attacks: list, epoch=1, device="cuda"):
    model.train()
    for i in range(epoch):
        for (x, y) in tqdm.tqdm(train_dataloader, desc=f"[INFO] ST&AT Epoch {i + 1}"):
            for attack in attacks:
                iter_step(model, x, y, optimizer, loss_fun, attack, device)
