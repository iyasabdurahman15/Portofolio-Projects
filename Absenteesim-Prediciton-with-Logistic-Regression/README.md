# Absenteeism-Prediction-with-Logistic-Regression
## Overview
Proyek ini bertujuan untuk membuat model machine learning untuk memprediksi Excessive Absenteeism (Absen Berlebihan) pada karyawan kantor dengan menggunakan Logistic Regression. Selain itu, juga dibuat module untuk memprediksi langsung data baru untuk memprediksi apakah karyawan tersebut akan mengalami absen berlebihan atau tidak.

## File Descriptions
- Absenteeism_data.csv
  Merupakan dataset yang akan digunakan untuk membangun model.
- Absenteesim_preprocessed.csv
  Merupakan dataset yang telah dilakukan preprocessing.
- Latihan 12.ipynb
  Merupakan file dimana dilakukan analisis, eksperimen serta preprocessing data.
- absenteeism_module.py
  Merupakan file untuk otomatisasi proses data baru berdasarkan proses preprocessing dan model yang telah dibangun sebelumnya.
- model_12
  Merupakan file model machine learning yang sebelumnya telah dilatih disimpan. File ini dipanggil pada file absenteeism_module.py
- scaler_12
  Merupakan file untuk melakukan proses scaling value data. File ini dipanggil pada file absenteeism_module.py
- Absenteeism_new_data.csv
  Merupakan dataset baru yang digunakan untuk memprediksi Excessive Absenteeism
- Absenteeism_Predictions.csv
  Merupakan hasil akhir prediksi model 
