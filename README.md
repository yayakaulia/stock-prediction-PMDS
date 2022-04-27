# stock-prediction-PMDS
project management pacmann ai - stock price prediction

1. Latar belakang (PRD: Stock Price Prediction)
Setiap hari nya, hampir semua perusahaan sekuritas, media bisnis, para analis memberikan prediksi harga saham. Walaupun tidak memberikan prediksi pada seluruh perusahaan yang listing (emiten), paling tidak pada perusahaan berkapitalisasi besar, atau yang menarik akhir-akhir ini. Paling pasti adalah Indeks harga gabungan akan naik atau turun ke nilai berapa. Apakah prediksi dari orang-orang atau lembaga ini tepat? Sayangnya tidak ada yang begitu memperhatikan. Karena antara kita tidak tahu metode dan perhitungan yang mereka gunakan, atau sekedar intuisi saja yang dimiliki setiap orang yang terjun. Lalu, bagaimana jika prediksi itu dilakukan dengan menggunakan metode Machine Learning? Itu yang akan kita cari tahu.

2. Workflow product
Rangkain penggunan produk yang saya bikin ini cukup mudah. Kita tinggal download salah satu emiten atau perusahaan listing di indonesia lewat yahoo finance. Setelah itu tidak perlu edit apapun pada file itu karena sudah disiapkan codingan untuk menyelaraskannya dalam algoritma ini. setelah itu runs untuk mendapatkan parameternya kemudian kita akan mendapatkan best params dengan input harga terakhir (dan fitur lain) dari emiten tersebut dan akan mengahasilkan prediksi untuk harga besok.

3. Penjelasan fitur
kita akan menggunakan fitur harga saham, volume saham, harga IHSG, dan volume IHSG. y predict kita adalah harga keesokan saham tersebut (harga + 1 hari index). Kemudian dari fitur harga saham akan dilakukan feature engineering seperti Moving Average dan jarak Moving Averange ke Harga saham. Untuk pemilihan model masih menggunakan berbagai bentuk dari Linear REggresion.

4. code
