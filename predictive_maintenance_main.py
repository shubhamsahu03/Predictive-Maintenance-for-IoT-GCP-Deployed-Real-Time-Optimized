import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud import storage
import warnings
warnings.filterwarnings('ignore')

class IoTSensorDataGenerator:
    def __init__(self, num_sensors=100, days=365):
        self.num_sensors = num_sensors
        self.days = days
        self.scaler = MinMaxScaler()

    def generate_synthetic_data(self):
        np.random.seed(42)

        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=self.days),
            end=datetime.now(),
            freq='H'
        )

        data = []
        for sensor_id in range(self.num_sensors):
            for timestamp in timestamps:

                base_temp = 25 + np.random.normal(0, 5)
                base_vibration = 0.5 + np.random.normal(0, 0.2)
                base_pressure = 1013 + np.random.normal(0, 50)
                base_humidity = 50 + np.random.normal(0, 15)
                base_current = 2.5 + np.random.normal(0, 0.5)

                degradation_factor = min(1.0, (timestamp - timestamps[0]).days / 365)

                if np.random.random() < 0.15 * degradation_factor:
                    temp_anomaly = np.random.choice([1.5, -1.5]) * np.random.exponential(10)
                    vibration_anomaly = np.random.exponential(2)
                    pressure_anomaly = np.random.choice([1, -1]) * np.random.exponential(100)
                    current_anomaly = np.random.exponential(1)
                    failure_label = 1
                else:
                    temp_anomaly = 0
                    vibration_anomaly = 0
                    pressure_anomaly = 0
                    current_anomaly = 0
                    failure_label = 0

                data.append({
                    'sensor_id': f'sensor_{sensor_id:03d}',
                    'timestamp': timestamp,
                    'temperature': base_temp + temp_anomaly,
                    'vibration': base_vibration + vibration_anomaly,
                    'pressure': base_pressure + pressure_anomaly,
                    'humidity': max(0, min(100, base_humidity)),
                    'current': base_current + current_anomaly,
                    'failure': failure_label
                })

        return pd.DataFrame(data)

class TimeSeriesPreprocessor:
    def __init__(self, sequence_length=24, overlap=12):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.scalers = {}

    def create_sequences(self, data, target_col='failure'):
        features = ['temperature', 'vibration', 'pressure', 'humidity', 'current']

        sequences = []
        labels = []

        for sensor_id in data['sensor_id'].unique():
            sensor_data = data[data['sensor_id'] == sensor_id].sort_values('timestamp')

            if sensor_id not in self.scalers:
                self.scalers[sensor_id] = MinMaxScaler()
                sensor_features = self.scalers[sensor_id].fit_transform(sensor_data[features])
            else:
                sensor_features = self.scalers[sensor_id].transform(sensor_data[features])

            sensor_labels = sensor_data[target_col].values

            step_size = self.sequence_length - self.overlap
            for i in range(0, len(sensor_features) - self.sequence_length + 1, step_size):
                sequence = sensor_features[i:i + self.sequence_length]
                label = int(np.any(sensor_labels[i:i + self.sequence_length]))

                sequences.append(sequence)
                labels.append(label)

        return np.array(sequences), np.array(labels)

    def save_scalers(self, filepath):
        joblib.dump(self.scalers, filepath)

    def load_scalers(self, filepath):
        self.scalers = joblib.load(filepath)

class PredictiveMaintenanceModel:
    def __init__(self, sequence_length=24, num_features=5):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.history = None

    def build_model(self):
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True,
                       input_shape=(self.sequence_length, self.num_features)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.flatten().tolist()
        }

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)

class GCPDeployment:
    def __init__(self, project_id, region='us-central1'):
        self.project_id = project_id
        self.region = region
        self.client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)

    def setup_bigquery_dataset(self, dataset_id='iot_sensor_data'):
        dataset_ref = self.client.dataset(dataset_id)

        try:
            self.client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_id} already exists")
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = self.client.create_dataset(dataset)
            print(f"Created dataset {dataset_id}")

        table_id = f"{self.project_id}.{dataset_id}.sensor_readings"

        schema = [
            bigquery.SchemaField("sensor_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("temperature", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("vibration", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("pressure", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("humidity", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("current", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("failure", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("prediction", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP", mode="NULLABLE")
        ]

        try:
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            print(f"Created table {table_id}")
        except:
            print(f"Table {table_id} already exists")

    def upload_data_to_bigquery(self, dataframe, table_id):
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=False
        )

        job = self.client.load_table_from_dataframe(
            dataframe, table_id, job_config=job_config
        )
        job.result()

        print(f"Loaded {len(dataframe)} rows to {table_id}")

    def deploy_model_to_vertex_ai(self, model_path, model_name):
        aiplatform.init(project=self.project_id, location=self.region)

        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=model_path,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest"
        )

        endpoint = model.deploy(
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=10
        )

        print(f"Model deployed to endpoint: {endpoint.resource_name}")
        return endpoint

class RealTimePredictor:
    def __init__(self, model_path, scalers_path):
        self.model = keras.models.load_model(model_path)
        self.preprocessor = TimeSeriesPreprocessor()
        self.preprocessor.load_scalers(scalers_path)
        self.sequence_buffer = {}

    async def process_sensor_event(self, sensor_data):
        sensor_id = sensor_data['sensor_id']
        features = ['temperature', 'vibration', 'pressure', 'humidity', 'current']

        if sensor_id not in self.sequence_buffer:
            self.sequence_buffer[sensor_id] = []

        feature_values = [sensor_data[feature] for feature in features]

        if sensor_id in self.preprocessor.scalers:
            scaled_features = self.preprocessor.scalers[sensor_id].transform([feature_values])[0]
        else:
            print(f"Warning: No scaler found for {sensor_id}")
            return None

        self.sequence_buffer[sensor_id].append(scaled_features)

        if len(self.sequence_buffer[sensor_id]) > 24:
            self.sequence_buffer[sensor_id].pop(0)

        if len(self.sequence_buffer[sensor_id]) == 24:
            sequence = np.array([self.sequence_buffer[sensor_id]])
            prediction = self.model.predict(sequence, verbose=0)[0][0]

            return {
                'sensor_id': sensor_id,
                'timestamp': sensor_data['timestamp'],
                'failure_probability': float(prediction),
                'alert': prediction > 0.5
            }

        return None

class ModelOptimizer:
    def __init__(self, model):
        self.model = model

    def quantize_model(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        quantized_model = converter.convert()
        return quantized_model

    def _representative_data_gen(self):
        for _ in range(100):
            yield [np.random.random((1, 24, 5)).astype(np.float32)]

    def prune_model(self, X_train, y_train):
        import tensorflow_model_optimization as tfmot

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )

        model_for_pruning = prune_low_magnitude(
            self.model,
            pruning_schedule=pruning_schedule
        )

        model_for_pruning.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep()
        ]

        model_for_pruning.fit(
            X_train, y_train,
            epochs=10,
            callbacks=callbacks,
            verbose=0
        )

        final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        return final_model

class MonitoringDashboard:
    def __init__(self, bigquery_client):
        self.client = bigquery_client

    def get_prediction_metrics(self, hours=24):
        query = f"""
        SELECT
            COUNT(*) as total_predictions,
            AVG(prediction) as avg_prediction,
            SUM(CASE WHEN prediction > 0.5 THEN 1 ELSE 0 END) as alerts_generated,
            COUNT(DISTINCT sensor_id) as sensors_monitored
        FROM `{self.client.project}.iot_sensor_data.sensor_readings`
        WHERE prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """

        results = self.client.query(query).result()
        return list(results)[0] if results.total_rows > 0 else None

    def get_sensor_health_summary(self):
        query = """
        SELECT
            sensor_id,
            COUNT(*) as readings_count,
            AVG(prediction) as avg_failure_probability,
            MAX(prediction_timestamp) as last_prediction
        FROM `{}.iot_sensor_data.sensor_readings`
        WHERE prediction IS NOT NULL
        GROUP BY sensor_id
        ORDER BY avg_failure_probability DESC
        LIMIT 20
        """.format(self.client.project)

        results = self.client.query(query).result()
        return [dict(row) for row in results]

async def main():
    print("Initializing Predictive Maintenance System...")

    data_generator = IoTSensorDataGenerator(num_sensors=50, days=180)
    print("Generating synthetic IoT sensor data...")
    raw_data = data_generator.generate_synthetic_data()

    preprocessor = TimeSeriesPreprocessor(sequence_length=24, overlap=12)
    print("Creating time series sequences...")
    X, y = preprocessor.create_sequences(raw_data)

    print(f"Generated {len(X)} sequences with shape {X.shape}")
    print(f"Failure rate: {np.mean(y):.3f}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    model = PredictiveMaintenanceModel(sequence_length=24, num_features=5)
    print("Building neural network model...")
    model.build_model()

    print("Training model...")
    training_start = datetime.now()
    history = model.train_model(X_train, y_train, X_val, y_val, epochs=30, batch_size=64)
    training_time = (datetime.now() - training_start).total_seconds()

    print(f"Training completed in {training_time:.2f} seconds")

    print("Evaluating model performance...")
    test_results = model.evaluate_model(X_test, y_test)

    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['classification_report']['1']['precision']:.4f}")
    print(f"Recall: {test_results['classification_report']['1']['recall']:.4f}")
    print(f"F1-Score: {test_results['classification_report']['1']['f1-score']:.4f}")

    model.save_model('predictive_maintenance_model.h5')
    preprocessor.save_scalers('sensor_scalers.pkl')

    print("Model saved successfully!")

    optimizer = ModelOptimizer(model.model)
    print("Optimizing model...")

    quantized_model = optimizer.quantize_model()
    with open('quantized_model.tflite', 'wb') as f:
        f.write(quantized_model)

    print("Model optimization completed!")

    print("\nSimulating real-time predictions...")
    predictor = RealTimePredictor('predictive_maintenance_model.h5', 'sensor_scalers.pkl')

    sample_events = raw_data.sample(10).to_dict('records')
    for event in sample_events:
        prediction = await predictor.process_sensor_event(event)
        if prediction:
            print(f"Sensor {prediction['sensor_id']}: Failure Probability = {prediction['failure_probability']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())