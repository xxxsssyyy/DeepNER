# encoding=utf8
import os
import codecs
import pickle
import itertools
import sys
from collections import OrderedDict
sys.path.append('../')

import tensorflow as tf
from models.Transformer_CRF import TransformerCRFModel
import numpy as np
from utils.loader import load_sentences, update_tag_scheme
from utils.loader import char_mapping, tag_mapping
from utils.loader import augment_with_pretrained, prepare_padding_dataset
from utils.utils import get_logger, make_path, clean, create_model, save_model
from utils.utils import print_config, save_config, load_config, test_ner
from utils.data_utils import load_word2vec, create_input, input_from_line_padding, BatchManager
from utils.metric import get_ner_fmeasure, recover_label


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("num_units",    100,        "Embedding size for characters")
flags.DEFINE_string("tag_schema",   "iob",      "tagging schema iobes or iob")
flags.DEFINE_integer("heads",       4,          "Num of multi-attention heads")
flags.DEFINE_integer("max_seq_len", 100,        "Num of max len of sequence")
flags.DEFINE_integer("num_blocks",  2,          "Num blocks")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.1,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training") 
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_string("save_path",    "save_transformer",      "Father path to save file")
flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    flags.FLAGS.save_path+"/ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     flags.FLAGS.save_path+"/maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  flags.FLAGS.save_path+"/config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  flags.FLAGS.save_path+"/result",       "Path for results")
flags.DEFINE_string("emb_file",     "../data/wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file", "../data/example.train",  "Path for train data")
flags.DEFINE_string("dev_file", "../data/example.dev",    "Path for dev data")
flags.DEFINE_string("test_file", "../data/example.test",   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["num_units"] = FLAGS.num_units
    config["num_tags"] = len(tag_to_id)

    config["batch_size"] = FLAGS.batch_size
    config["heads"] = FLAGS.heads
    config["max_seq_len"] = FLAGS.max_seq_len
    config["num_blocks"] = FLAGS.num_blocks

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    #update_tag_scheme(train_sentences, FLAGS.tag_schema)
    #update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        os.makedirs('%s' % FLAGS.save_path)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_padding_dataset(
        train_sentences, FLAGS.max_seq_len, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_padding_dataset(
        dev_sentences, FLAGS.max_seq_len, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_padding_dataset(
        test_sentences, FLAGS.max_seq_len, char_to_id, tag_to_id, FLAGS.lower
    )
    
    
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    """
    batch = train_manager.batch_data[0]
    strings, chars, segs, tags = batch
    for chrs in chars:
        print(chrs)
    for chrs in segs:
        print(chrs)
    print(tag_to_id)
    """
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)
    
    log_path = os.path.join(FLAGS.save_path, "log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = TransformerCRFModel(config, is_training=True)
        sess.run(tf.global_variables_initializer())
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            predict_lists = []
            source_tag = []
            best_dev_f1 = 0.0
            best_test_f1 = 0.0
            for batch in dev_manager.iter_batch(shuffle=False):
                lengths, logits = model.run_step(sess, False, batch)
                _, chars, segs, tags = batch
                transition = model.transition.eval(session=sess)
                pre_seq = model.predict(logits, transition, lengths)
                pre_label = recover_label(pre_seq, lengths, id_to_tag)
                """
                for p in range(len(pre_label)):
                    print(chars[p])
                    print(pre_label[p])
                """
                source_label = recover_label(tags, lengths, id_to_tag)
                predict_lists.extend(pre_label)
                source_tag.extend(source_label)
            train_loss_v = np.round(float(np.mean(loss)), 4)
            print('****************************************************')
            acc, p, r, f = get_ner_fmeasure(source_tag, predict_lists, config["tag_schema"])
            logger.info('epoch:\t{}\ttrain loss:\t{}\t'.format(i+1, train_loss_v))
            logger.info('dev acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
            
            
            for batch in test_manager.iter_batch(shuffle=False):
                lengths, logits = model.run_step(sess, False, batch)
                _, chars, segs, tags = batch
                transition = model.transition.eval(session=sess)
                pre_seq = model.predict(logits, transition, lengths)
                pre_label = recover_label(pre_seq, lengths, id_to_tag)
                source_label = recover_label(tags, lengths, id_to_tag)
                predict_lists.extend(pre_label)
                source_tag.extend(source_label)

            acc_t, p_t, r_t, f_t = get_ner_fmeasure(source_tag, predict_lists, config["tag_schema"])
            logger.info('test acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc_t, p_t, r_t, f_t))
            if f > best_dev_f1:
                save_model(sess, model, FLAGS.ckpt_path, logger)
                best_dev_f1 = f
                best_test_f1 = f_t
                logger.info('save epoch:\t{} model with best dev f1-score'.format(i+1))
            
            print('****************************************************\n\n')

            


def main(_):
    FLAGS.train = True
    FLAGS.clean = True
    clean(FLAGS)
    train()
    
    #evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)