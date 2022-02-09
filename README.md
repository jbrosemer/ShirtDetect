Program uses OpenCV and HaarCascade to identify someone's face, after which finds a region below their face which we expect to be the shirt.
After detecting the shirt region separates that into a new image and uses a Kmeans clustering method from Tim Poulsen
https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html
to find the most common color on the shirt, and then classifies that as one of the colors provided, could add more colors in the future.
