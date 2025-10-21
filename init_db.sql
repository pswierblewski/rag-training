-- Create database
CREATE DATABASE rag_db_training;

-- Connect to the database
\c rag_db_training;

-- Create sentences table
CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL
);

-- Insert sample Polish sentences
INSERT INTO sentences (sentence) VALUES
('Dzień dobry, jak się masz?'),
('Pogoda dzisiaj jest piękna i słoneczna.'),
('Kocham czytać książki w wolnym czasie.'),
('Polska to kraj w Europie Środkowej.'),
('Warszawa jest stolicą Polski.'),
('Lubię pić kawę rano.'),
('Gdzie jest najbliższa apteka?'),
('Uczę się programowania w Pythonie.'),
('Kolacja jest gotowa o godzinie osiemnastej.'),
('Mój ulubiony kolor to niebieski.'),
('W weekendy lubię jeździć na rowerze.'),
('Jak dojść do dworca kolejowego?'),
('Jutro mam ważne spotkanie w biurze.'),
('Moja rodzina pochodzi z Krakowa.'),
('Chciałbym nauczyć się grać na gitarze.'),
('Sklep jest otwarty od poniedziałku do soboty.'),
('Lubię oglądać filmy w kinie.'),
('Czy możesz mi pomóc z tym problemem?'),
('Zima w Polsce może być bardzo mroźna.'),
('Studiuję informatykę na uniwersytecie.'),
('Mój pies nazywa się Max.'),
('Wolę herbatę od kawy.'),
('Kraków to jedno z najpiękniejszych miast w Polsce.'),
('Czy możesz podać mi sól?'),
('Moja siostra pracuje jako lekarz.'),
('Lubię gotować włoskie dania.'),
('W czerwcu jadę na wakacje nad morze.'),
('Potrzebuję kupić nowy telefon.'),
('Uczę się języka angielskiego.'),
('Moja córka idzie do szkoły podstawowej.'),
('Pracuję jako programista w firmie IT.'),
('Czy ten autobus jedzie do centrum?'),
('Wieczorem lubię czytać książki.'),
('Mój brat studiuje medycynę.'),
('Czy mógłbyś zamknąć okno?'),
('Lubię jeść pierogi z kapustą i grzybami.'),
('W Polsce mamy cztery pory roku.'),
('Jutro będzie padać deszcz.'),
('Muszę iść do lekarza na badania.'),
('Czy ten sklep przyjmuje karty kredytowe?'),
('Mój samochód jest w warsztacie.'),
('Lubię słuchać muzyki klasycznej.'),
('Wieczorem idę na trening fitness.'),
('Moja mama gotuje najlepsze zupy.'),
('Czy możesz mi powiedzieć, która godzina?'),
('W sobotę spotykam się z przyjaciółmi.'),
('Muszę zapłacić rachunki za prąd.'),
('Lubię spacerować po parku.'),
('Mój ulubiony pisarz to Henryk Sienkiewicz.'),
('Czy możesz mi polecić dobrą restaurację?'),
('Pracuję zdalnie z domu.'),
('Moja babcia mieszka na wsi.'),
('Lubię jeździć na nartach zimą.'),
('Czy ten hotel ma Wi-Fi?'),
('Muszę kupić bilety na koncert.'),
('Studiuję sztuczną inteligencję.'),
('Mój kot lubi spać na kanapie.'),
('Czy możesz mówić wolniej?'),
('Lubię zwiedzać muzea i galerie sztuki.'),
('W Polsce mamy wiele pięknych zamków.'),
('Muszę oddać książki do biblioteki.'),
('Czy mógłbyś wyłączyć światło?'),
('Mój dziadek był inżynierem.'),
('Lubię pływać w basenie.'),
('Czy możesz mi pokazać drogę?');



