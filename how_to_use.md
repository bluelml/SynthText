### genreate training data for CRNN 
run  
```
python gen.py
```
will generate data and save in folder bib_data.


### how to add new fonts
```
1. cp *.ttf file to ./data/fonts
2. update the file ./data/fonts/fontlist.txt
3. run 'python invert_font_size.py'
``` 
