import split_folders

split_folders.ratio('C:\\Users\\Piotr\\Documents\\Studia\\Informatyka PW\\2 semestr\\SNR\\input\\images\\Images',
                    'C:\\Users\\Piotr\\Documents\\Studia\\Informatyka PW\\2 semestr\\SNR\\input\\images\\dataset',
                    seed=13,
                    ratio=(.8, .1, .1)
                    )
