# EMNIST Əlyazma Tanıma Tətbiqi

EMNIST dataseti ilə təlim keçmiş CNN modeli istifadə edərək əlyazma rəqəm və hərfləri tanıyan veb tətbiq.

## Layihə Haqqında

Bu layihə istifadəçilərin canvas üzərində yazdığı rəqəm və hərfləri maşın öyrənməsi vasitəsilə tanıyır. Model 62 fərqli simvolu tanıya bilir: 0-9 rəqəmləri, A-Z böyük hərflər və a-z kiçik hərflər.

## Qovluq Strukturu

emnist-recognizer/
│
├── backend/
│   ├── app.py           - Flask server
│   ├── utils.py         - Şəkil emalı funksiyaları
│   └── train.py         - Model təlim kodu
│
├── frontend/
│   └── index.html       - Veb interfeys
│
├── model/
│   └── emnist_cnn.h5    - Təlim keçmiş model
│
├── dataset/
│   ├── emnist-byclass-train.csv
│   └── emnist-byclass-test.csv
│
├── requirements.txt     - Python-ın pip edilən kitabxanaları
└── README.md           



### 1. Model Təlimi

CNN modeli EMNIST ByClass dataseti üzərində təlim keçib.

Model rəqəmlər (0-9), böyük hərflər (A-Z) və kiçik hərfləri (a-z) fərqləndirə bilir.

Təlim keçmiş model .h5 formatında ixrac edilib.

### 2. Veb Tətbiq

Backend: Flask istifadə edilib, model API vasitəsilə UI-ya qoşulub.

Frontend: Vanilla JavaScript və HTML5 Canvas ilə hazırlanıb.

Buttonlar: "Predict" və "Clear" düymələri əlavə edilib.

Vizuallaşdırma: Top-3 təxminlər bar chart ilə göstərilir.

### 3. Şəkil Emalı

İstifadəçi çəkilişləri 28x28 ölçüyə çevrilir və boz tona salınır.

EMNIST formatına uyğunlaşdırmaq üçün 180 dərəcə fırlatma tətbiq edilir.

Simvol mərkəzləşdirilir.

## Quraşdırma

1. Repozitoriyanı klonlayın:
bash
git clone https://github.com/royaefendiyeva/emnist-recognizer.git
cd emnist-recognizer


2. Lazımi kitabxanaları quraşdırın:
bash
pip install -r requirements.txt


3. EMNIST datasetini yükləyin və dataset qovluğuna yerləşdirin.

4. Modeli təlim edin:
bash
python backend/train.py


5. Flask serveri işə salın:
bash
python backend/app.py


6. Brauzerdə açın: http://127.0.0.1:5000

## İstifadə

Canvas üzərində rəqəm və ya hərf yazın.

"Predict" düyməsinə basın.

Nəticələri görün - model 3 ən yüksək ehtimallı cavabı göstərir.

"Clear" ilə təmizləyin və yenidən cəhd edin.
