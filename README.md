# egg-dimensions

Measure *Drosophila* egg dimensions from images using OpenCV

* Install dependencies by running the shell script `install-dependencies.sh`:
```
./install-dependencies.sh
```
* Put images of the eggs to measured in the `toMeasure` directory

* Run the python script `measure-eggs.py`:
```
python3 measure-eggs.py
```

* The script would output all the names of all the images in `toMeasure` directory as list and ask for the scale of the images. Input a number corresponding to the scale of the images.

* Note: The images should contain a scale indicator on the left-hand-side.

![](Image_processing.gif)
