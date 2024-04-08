import regex as re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

main_data = pd.read_csv(r"C:\Users\ciaun\Desktop\dataset.csv")

df = main_data.copy()  # We take a copy of the original data incase we needed the original data later
df.dropna(axis=1, how='all', inplace=True)  # Dropping rows which are all NaN
df.fillna(0, inplace=True)  # Replacing the NaN with 0


# Creating a custom label encoder so we can specify which number the encoding starts from
class CustomLabelEncoder(LabelEncoder):
    def __init__(self, start=0):
        self.start = start
        super().__init__()

    def fit_transform(self, y):
        encoded = super().fit_transform(y)
        encoded += self.start
        return encoded


# Flatten the 'Disease' column into a single Series
flattened_series = df['Disease'].astype(str)

# Create and fit label encoder for the 'Disease' column
encoder = CustomLabelEncoder(start=200)

encoded_values = encoder.fit_transform(flattened_series)
df['Disease'] = encoded_values

mapping_data = {'label_encoder': encoder}

# Saving the mapping of the label column "Disease" to use later
label_mapping = {k: v for k, v in zip(mapping_data['label_encoder'].classes_,
                                      range(200, 200 + len(mapping_data['label_encoder'].classes_)))}

df.head()

# Stack the entire data into a single Series.
# We are stacking the entire data because there're similar values in different columns. **REMEMBER THIS**
encode_df = df.copy()  # Again, taking a copy because we might need the original later.
encode_df = encode_df.drop(["Disease"], axis=1)
flattened_series = encode_df.stack().astype(str)

# Create and fit label encoder.
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(flattened_series)

# Reshape the encoded values back to the original DataFrame shape.
F_encoded_df = pd.DataFrame(encoded_values.reshape(encode_df.shape), columns=encode_df.columns,
                            index=encode_df.index)

# Store the mapping data for future use
Fmapping_data = {'label_encoder': encoder}
feature_mapping = {k: v for k, v in zip(Fmapping_data['label_encoder'].classes_,
                                        Fmapping_data['label_encoder']. \
                                        transform(Fmapping_data['label_encoder'].classes_))}
F_encoded_df.head(3)

label_encoded_df = pd.concat([df['Disease'], F_encoded_df], axis=1)
label_encoded_df.head()

# Creating X and y
model_features = label_encoded_df.columns.tolist()
model_features.remove("Disease")
X = label_encoded_df[model_features]
y = label_encoded_df["Disease"]

y_encoded = pd.get_dummies(y)

# Reshape the data
X_reshaped = X.values.reshape(-1, 1)
scaler = StandardScaler().fit(X_reshaped)
X_scaled_reshaped = scaler.transform(X_reshaped)
# Reshape back to original shape
X_scaled = X_scaled_reshaped.reshape(X.shape)
X_df = pd.DataFrame(X_scaled)
X_df.head()

X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.25, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
X_eval_tensor = tf.convert_to_tensor(X_eval.values, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float64)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float64)
y_eval_tensor = tf.convert_to_tensor(y_eval, dtype=tf.float64)

def train():
    with tf.device('/GPU:0'):
        model_1 = keras.Sequential([
            layers.Input(shape=(X_train_tensor.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='tanh'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(y_train_tensor.shape[1], activation='softmax')])

        model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max')
        history = model_1.fit(X_train_tensor, y_train_tensor, epochs=500, callbacks=[early_stopping],
                              batch_size=16, validation_data=(X_eval_tensor, y_eval_tensor))

    model_1.evaluate(X_test_tensor, y_test_tensor)
    model_1.save("model_1.keras")

def encode_user_input(user_input, mapping=feature_mapping):
    '''
    This function takes user input and transform it to the same encoding
    the original data, which the model was trained on, has.

    Args:
        user_input (str): The user input.
        mapping (dict): The mapping the label_encoder used earlier.

    Returns:
        str: encoded user input.
    '''
    encoded_input = []
    for symptom in user_input:
        for key in mapping.keys():
            if symptom.strip().lower() == key.strip().lower():
                encoded_input.append(mapping[key])
                break  # Break out of inner loop if a match is found
    return encoded_input


def predict(user_input):
    model_1 = tf.keras.models.load_model('model_1.keras')

    encoded_input = encode_user_input(user_input=user_input)

    input_tensor = tf.cast(encoded_input, tf.float32)


    padding_value = tf.constant(130, dtype=tf.float32)
    desired_length = X_train_tensor[1].shape[0]
    padding_length = desired_length - tf.shape(input_tensor)[0]
    padding_tensor = tf.fill((padding_length,), padding_value)
    final_input = tf.concat([input_tensor, padding_tensor], axis=0)
    target_index = y_encoded.columns.tolist() # If you remember, the column names after the one-hot-encoding ARE the mapping of the target values.
    final_array = final_input.numpy()
    final_reshaped = final_array.reshape(-1, 1)
    X_scaled = scaler.transform(final_reshaped)
    final_tensor = tf.convert_to_tensor(X_scaled)
    final_tensor = tf.squeeze(final_tensor)

    predict_proba = model_1.predict(tf.expand_dims(final_input, axis = 0)) # Expanding dims to get (1,17)
    predicted_class_index = np.argmax(predict_proba) # Getting the 'index' of our prediction
    prediction_encode = target_index[predicted_class_index] # Getting to mapping of that 'index' using y column names
    inverse_label_encoding = {v: k for k, v in label_mapping.items()} # Inverse the label encoding
    prediction = inverse_label_encoding[prediction_encode]
    print(prediction)

# df = main_data.copy() # As usual, taking a copy from that data incase we needed the original later
# # Combine all symptom columns into a single column
# df['All Symptoms'] = df.apply(lambda row: ','.join(row.dropna()), axis=1)
# # Drop duplicate symptoms within each cell
# df['All Symptoms'] = df['All Symptoms'].apply(lambda x: ','.join(sorted(set(x.split(','))) if x else ''))
# stay_cols= ['Disease', 'All Symptoms']
# df = df[stay_cols]
# df.head()
#
# def strip_to_basic_tokens(text):
#     # Remove doble spaces and underscores
#     text = re.sub(r'[_\s]+', ' ', text)
#     # Split by commas and lowercase the tokens
#     tokens = [token.strip().lower() for token in text.split(',')]
#     return tokens
#
# # Apply the function to 'All Symptoms' column
# df['Basic Tokens'] = df['All Symptoms'].apply(strip_to_basic_tokens)
# df['Basic Tokens'] = df['Basic Tokens'].apply(lambda x: ', '.join(x))
# df = df.drop(['All Symptoms'], axis = 1)
# df.head()
#
# dfE = df.copy() # Taking a copy because we never know what might happen
# dfE['Basic Tokens'] = dfE['Basic Tokens'].apply(lambda x: x.split(', '))
#
# mlb = MultiLabelBinarizer()
# # Fit and transform the 'Basic Tokens' column
# one_hot_encoded = pd.DataFrame(mlb.fit_transform(dfE['Basic Tokens']), columns=mlb.classes_, index=df.index)
#
# # Concatenate the one-hot encoded DataFrame with the original DataFrame
# df_encoded = pd.concat([dfE, one_hot_encoded], axis=1)
#
# # Drop the 'Basic Tokens' column
# df_encoded = df_encoded.drop(columns=['Basic Tokens'])
# df_encoded.head()
#
# disease_names = [key for key in label_mapping.keys()]
# diseases = [strip_to_basic_tokens(disease) for disease in disease_names]
# diseases_cleaned = [item[0] if isinstance(item, list) else item for item in diseases]
# df_encoded = df_encoded.drop(diseases_cleaned, axis=1)
#
# model_features = df_encoded.columns.tolist()
# model_features.remove("Disease")
# X = df_encoded[model_features]
# y = df_encoded["Disease"]
# y_encoded = pd.get_dummies(y)
#
# X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
# X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
# X_eval_tensor = tf.convert_to_tensor(X_eval.values, dtype=tf.float32)
# y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float64)
# y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float64)
# y_eval_tensor = tf.convert_to_tensor(y_eval, dtype=tf.float64)
# #
# with tf.device('/GPU:0'):
#     model_2 = keras.Sequential([
#         layers.Input(shape=(X_train_tensor.shape[1],)),
#         layers.Dense(160, activation='relu'),
#         layers.Dropout(0.1),
#         layers.Dense(200, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(240, activation='tanh'),
#         layers.BatchNormalization(),
#         layers.Dense(240, activation='tanh'),
#         layers.Dropout(0.2),
#         layers.Dense(200, activation='relu'),
#         layers.Dropout(0.1),
#         layers.Dense(160, activation='relu'),
#         layers.Dense(y_train_tensor.shape[1], activation='softmax')])
#
#     model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max')
#     history = model_2.fit(X_train_tensor, y_train_tensor, epochs=500, callbacks=[early_stopping],
#                           batch_size=16, validation_data=(X_eval_tensor, y_eval_tensor))

# If you remember in the first model, we took a row from the origial data to test the model
# We aren't going to do this here, let's REALLY test it


if __name__ == "__main__":
    # train()
    predict(['back_pain', 'neck_pain', 'dizziness', 'loss_of_balance'])
