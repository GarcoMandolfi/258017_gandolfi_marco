def init():
    global DEVICE
    DEVICE = 'cuda:0'

    global hidden_size
    hidden_size = 200

    global embedding_size
    embedding_size = 300

    global lr
    lr = 0.1

    global clip_size
    clip_size = 5