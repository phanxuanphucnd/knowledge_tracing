import numpy

from anivia.utils.data_utils import *
from anivia.model import AKTModel
from anivia.dataset import AKTDataset
from anivia.learner import AKTLearner


def test_train():
    data_df = load_pickle(data_path='./data/train-subset.pkl')

    # n_question = get_n_question(data_df=data_df)
    group = group_by_user_id(data_df=data_df)
    train_group, test_group = train_test_split(data_df=data_df, pct=0.1)

    ## define paramerters
    n_question = 13523
    max_seq = 200
    n_pid = -1
    n_blocks = 1
    d_model = 256
    dropout = 0.05
    kq_same = 1
    n_heads = 8
    d_ff = 2048
    l2 = 1e-5
    final_fc_dim = 512

    batch_size = 24
    learning_rate = 1e-5
    max_learning_rate = 2e-3
    n_epochs = 1

    train_dataset = AKTDataset(train_group, n_question=n_question, max_seq=max_seq)
    test_dataset = AKTDataset(test_group, n_question=n_question, max_seq=max_seq)

    model = AKTModel(
        n_question=n_question, n_pid=n_pid, n_blocks=n_blocks,
        d_model=d_model, n_heads=n_heads, dropout=dropout,
        kq_same=kq_same, model_type='akt', l2=l2,
        final_fc_dim=final_fc_dim, d_ff=d_ff
    )

    learner = AKTLearner(model=model)
    learner.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        max_learning_rate=max_learning_rate,
        save_path='./models',
        model_name='akt_model'
    )

def test_inference():
    data_df = load_pickle(data_path='./data/train-subset.pkl')

    group = group_by_user_id(data_df=data_df)

    n_question = 13523
    max_seq = 200
    n_pid = -1
    n_blocks = 1
    d_model = 256
    dropout = 0.05
    kq_same = 1
    n_heads = 8
    d_ff = 2048
    l2 = 1e-5
    final_fc_dim = 512

    model = AKTModel(
        n_question=n_question, n_pid=n_pid, n_blocks=n_blocks,
        d_model=d_model, n_heads=n_heads, dropout=dropout,
        kq_same=kq_same, model_type='akt', l2=l2,
        final_fc_dim=final_fc_dim, d_ff=d_ff
    )    

    learner = AKTLearner(model=model)
    learner.load_model('./models/akt_model.pt')

    output = learner.infer(group, n_question, user_id=46886, content_id=471)

    print(output)

test_train()
test_inference()