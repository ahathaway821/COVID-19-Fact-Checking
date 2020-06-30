import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split

def model(x_train, y_train, x_test, y_test):
    hub_model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(hub_model, output_shape=[20], input_shape=[], 
                           dtype=tf.string, trainable=True)
    
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)

    return model


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    df = pd.read_csv(os.path.join(args.train,'claims.csv'), index_col=0, engine="python")

    train_df, test_df = train_test_split(df, test_size=0.8, random_state=0)
    train_examples = train_df['claim']
    train_labels = train_df['label_binary']

    test_examples = test_df['claim']
    test_labels = test_df['label_binary']
    use_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        use_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')