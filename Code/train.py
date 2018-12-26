from model import *
from update import adam


def check_accuracy(model, x, y, batch_size=100):
    accuary = 0
    correct_num = 0
    N = x.shape[0]
    num_batches = N // batch_size
    if N % batch_size != 0:
        num_batches += 1
    for i in range(num_batches):
        st = i * batch_size
        ed = (i + 1) * batch_size
        result = model.forward(x[st:ed])
        scores = softmax(result)
        top_three = np.argsort(scores)[:, -3:]
        correct_num += np.sum((y - 1)[st:ed, None] == top_three)
        
    accuary = correct_num / N
    return accuary


def train(model, data, batch_size=100, num_epochs=10, lr_decay=0.95, verbose=True, print_every=100):

    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']

    num_train = x_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size , 1)
    num_iterations = num_epochs * iterations_per_epoch
    
    
    #initialization
    epoch = 0
    adam_params = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    #train
    for i in range(num_iterations):
        #One step, SGD
        batch_mask = np.random.choice(num_train, batch_size)
        loss, grads = model.itertaion(x_train[batch_mask], y_train[batch_mask])
        loss_history.append(loss)

        #print training loss 
        if verbose and i % print_every == 0:
            print('Iteration %d/%d, loss: %f' % (i+1, num_iterations, loss_history[-1]))            

        #Update parameters
        for p, w in model.params.items():
            dw = grads[p]
            model.params[p], adam_params[p] = adam(w,dw,adam_params[p])

        #end of one epoch
        epoch_end = (i + 1) % iterations_per_epoch == 0
        if epoch_end:
            epoch += 1
            #decay the learning rate
            for p in adam_params:
                adam_params[p]['learning_rate'] *= lr_decay
            #check accuary
            train_acc = check_accuracy(model, x_train, y_train)
            val_acc = check_accuracy(model, x_val, y_val)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if verbose:
                print('Epoch %d/%d, train_acc: %f, val_acc: %f'%(
                    epoch + 1, num_epochs, train_acc, val_acc))

        

