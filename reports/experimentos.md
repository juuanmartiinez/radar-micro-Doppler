# EXPERIMENTOS SOBRE MEDICIONES DE MODELOS

## 1) CNN desde cero:

    Precisión test: 0.9061371841155235
    [[48  0  0  0  0  0]
    [ 4 44  0  0  0  0]
    [ 0  0 48  0  0  0]
    [ 1  0  1 42  2  3]
    [ 0  2  1 11 34  0]
    [ 1  0  0  0  0 35]]
                precision    recall  f1-score   support

        andar      0.889     1.000     0.941        48
     sentarse      0.957     0.917     0.936        48
   levantarse      0.960     1.000     0.980        48
    agacharse      0.792     0.857     0.824        49
        beber      0.944     0.708     0.810        48
       caerse      0.921     0.972     0.946        36

     accuracy                          0.906       277
    macro avg      0.911     0.909     0.906       277
 weighted avg      0.910     0.906     0.904       277

# 2a) ResNet18 congelando parametros:

    Precisión test: 0.8303249097472925
    [[46  0  1  0  0  1]
    [ 2 37  4  4  1  0]
    [ 0  0 41  3  3  1]
    [ 3  1  3 39  3  0]
    [ 1  3  2 10 32  0]
    [ 0  0  0  0  1 35]]
                precision    recall  f1-score   support

        andar      0.885     0.958     0.920        48
     sentarse      0.902     0.771     0.831        48
   levantarse      0.804     0.854     0.828        48
    agacharse      0.696     0.796     0.743        49
        beber      0.800     0.667     0.727        48
       caerse      0.946     0.972     0.959        36

        accuracy                          0.830       277
       macro avg      0.839     0.836     0.835       277
    weighted avg      0.834     0.830     0.829       277

# 2b) ResNet18 con fine tuning:



Precisión test: 0.9530685920577617
[[48  0  0  0  0  0]
 [ 0 48  0  0  0  0]
 [ 0  0 47  0  1  0]
 [ 0  1  2 45  1  0]
 [ 0  1  0  7 40  0]
 [ 0  0  0  0  0 36]]
              precision    recall  f1-score   support

       andar      1.000     1.000     1.000        48
    sentarse      0.960     1.000     0.980        48
  levantarse      0.959     0.979     0.969        48
   agacharse      0.865     0.918     0.891        49
       beber      0.952     0.833     0.889        48
      caerse      1.000     1.000     1.000        36

    accuracy                          0.953       277
   macro avg      0.956     0.955     0.955       277
weighted avg      0.954     0.953     0.953       277

