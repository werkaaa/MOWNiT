import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import cv2

class OCR:
    def __init__(self, font, line_height=27, space_coefficient=0.35):
        self.letters_order = ['a', 'b', 'd', 'm', 'g', 'q', 'p', 'e',
                              'j', 'k', 'f', 'h', '0', 'n', 'o',
                              'c',  's', 'u', 'y', 'w', 'v', 'x',
                              'z', '1', '7', '4', 't', 'r', 'l', 'i', '2', '8',
                              '5', '6', '3', '9', '?', '!', ',', '.']

        self.letters = {}
        self.font = font
        self.medium_width = 0.
        self.letters_occurrences = {}
        self.text_reconstructed = None
        self.image_state = None
        for letter in self.letters_order[:-4]:
            self.letters[letter] = self.load_letter(f'{font}/{letter}.png')
            self.medium_width += self.letters[letter].shape[1]
            self.letters_occurrences[letter] = 0

        self.medium_width /= 36.
        self.space_coefficient = space_coefficient

        self.letters['?'] = self.load_letter(f'{font}/question-mark.png')
        self.letters_occurrences['?'] = 0
        self.letters['.'] = self.load_letter(f'{font}/dot.png')
        self.letters_occurrences['.'] = 0
        self.letters[','] = self.load_letter(f'{font}/comma.png')
        self.letters_occurrences[','] = 0
        self.letters['!'] = self.load_letter(f'{font}/exclamation-mark.png')
        self.letters_occurrences['!'] = 0

        self.line_height = line_height
        self.all_letters_path = f'{font}/all.png'
        self.rotation = False
        self.noise = False

    def zero_letters_occurrences(self):
        for key in self.letters_occurrences.keys():
            self.letters_occurrences[key] = 0

    def sensitivity(self, letter):
        if self.rotation and self.font == 'serif':
            less_sensitive = ['i', 'v', 'w', ',', '.', 'g']
            more_sensitive = ['b', 'o']
            return 0.85 if letter in less_sensitive else (0.92 if letter in more_sensitive else 0.9)
        elif (self.rotation and self.font == 'sans-serif') or self.noise:
            much_less_sensitive = ['i', '8', '.', 'g']
            less_sensitive = ['v', 'w', ',', 'l', 'r', '4', 'j']
            more_sensitive = ['1', 'o']
            return 0.84 if letter in much_less_sensitive else (0.89 if letter in less_sensitive else (0.94 if letter in more_sensitive else 0.9))
        else:
            less_sensitive = ['i', 'v', 'w', ',', '.', 'g']
            more_sensitive = ['b', 'o']
            return 0.85 if letter in less_sensitive else (0.92 if letter in more_sensitive else 0.9)


    def load_image(self, path):
        return np.array(ImageOps.invert(Image.open(path).convert('L')))

    def load_letter(self, path):
        image = self.load_image(path)
        rows, cols = np.ix_((image > 0).any(1), (image > 0).any(0))
        return image[max(0, np.min(rows)-1):min(image.shape[0], np.max(rows)+2), max(0, np.min(cols)):min(image.shape[1], np.max(cols)+1)]



    def filter_picks(self, image, sensitivity=0.9):
        image_filtered = np.copy(image)
        image_filtered[image < np.max(image) * sensitivity] = 0.
        return image_filtered

    def correlation(self, pattern, image):
        pattern_rotated = np.rot90(pattern, 2)
        frequencies_pattern = np.fft.fft2(pattern_rotated, image.shape)
        frequencies_image = np.fft.fft2(image)
        return np.real(np.fft.ifft2(np.multiply(frequencies_pattern, frequencies_image)))

    def find_letter_placement(self, letter, pattern, image):
        placements = set()
        correlation = self.correlation(pattern, image)
        filtered_correlation = self.filter_picks(correlation, self.sensitivity(letter))
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                if filtered_correlation[i][j] != 0.:
                    placements.add((i // self.line_height, j))
                    image[i - pattern.shape[0]:i + 1, j - pattern.shape[1]:j + 1] = 0

        return placements

    def dict_to_string(self, location_dict):
        all_letters_location = []
        for key in location_dict.keys():
            for l in list(location_dict[key]):
                all_letters_location.append((l[0], l[1], key))
        all_letters_location.sort()
        output = ""
        start_saving = False
        for i, letter in enumerate(all_letters_location):
            if all_letters_location[i - 1][0] < letter[0]:
                if start_saving:
                    output += '\n'
                else:
                    start_saving = True
            elif letter[1] - all_letters_location[i - 1][1] < 0.5 * self.letters[letter[2]].shape[1]:
                continue
            elif (letter[1] - self.letters[letter[2]].shape[1] - all_letters_location[i - 1][1]
                  > self.space_coefficient * self.medium_width):
                if start_saving:
                    output += ' '
            if start_saving:
                self.letters_occurrences[letter[2]] += 1
                output += letter[2]

        return output

    def get_concatenated_image(self, image1_path, image2_path):
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        dst = Image.new('L', (max(image1.width, image2.width), image1.height + image2.height), 255)
        dst.paste(image1, (0, 0))
        dst.paste(image2, (0, image1.height))
        return np.array(ImageOps.invert(dst.convert('L')))

    def image_to_text(self, path, rotated=False, noise=0.0):
        if rotated:
            image = self.correct_rotation(path)
            self.rotation = True
        else:
            image = self.get_concatenated_image(self.all_letters_path, path)

        if noise != 0.0:
            self.noise = True
            image = self.delete_noise(image, noise)

        letters_placements = {}
        self.image_state = [np.copy(image)]
        self.zero_letters_occurrences()

        for letter in self.letters_order:
            letters_placements[letter] = self.find_letter_placement(letter, self.letters[letter], image)
            self.image_state.append(np.copy(image))
        self.text_reconstructed = self.dict_to_string(letters_placements)
        return self.text_reconstructed

    def show_statistics(self, text):
        if self.text_reconstructed is None:
            print("No reconstructed text!")
            return

        else:
            text_stat = Counter(text)
            for letter in self.letters_order:
                print(f'{letter} occurred in text {text_stat[letter] if letter in text_stat.keys() else 0} times; '
                      f'found {self.letters_occurrences[letter] if letter in self.letters_occurrences.keys() else 0} in image')

            edit_distance = Levenshtein_Distance(text, self.text_reconstructed)
            print(f'Reconstruction correctness: {100*(len(text)-edit_distance)/len(text)}%')

    def get_letters_occurrences(self):
        return self.letters_occurrences

    def get_image_states(self):
        return self.image_state

    def correct_rotation(self, path):
        image = self.load_image(path)
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angle = round(angle)

        height, width = image.shape
        center = (width//2, height//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rows, cols = np.ix_((rotated > 0).any(1), (rotated > 0).any(0))
        rotated = rotated[max(0, np.min(rows)-4):min(rotated.shape[0], np.max(rows)+5), max(0, np.min(cols)-5):min(rotated.shape[1], np.max(cols)+5)]
        image2 = ImageOps.invert(Image.fromarray(rotated.astype(np.uint8)).convert('L'))
        image1 = Image.open(self.all_letters_path)
        dst = Image.new('L', (max(image1.width, image2.width), image1.height + image2.height), 255)
        dst.paste(image1, (0, 0))
        dst.paste(image2, (0, image1.height))

        return np.array(ImageOps.invert(dst.convert('L')))

    def delete_noise(self, image, fraction=0.3):
        im_fft = np.fft.fft2(image)
        r, c = im_fft.shape
        im_fft[int(r * fraction):int(r * (1 - fraction))] = 0
        im_fft[:, int(c * fraction):int(c * (1 - fraction))] = 0
        return np.abs(np.fft.ifft2(im_fft).real)


def Levenshtein_Distance(word1, word2):
    dist_matrix = [[0] * (len(word2) + 1) for i in range(len(word1) + 1)]

    for j in range(1, 1 + len(word2)):
        for i in range(1, 1 + len(word1)):
            if word1[i - 1] == word2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            dist_matrix[i][j] = min(dist_matrix[i - 1][j]+ 1, dist_matrix[i][j - 1] + 1,
                               dist_matrix[i - 1][j - 1] + substitution_cost)


    return dist_matrix[-1][-1]


# ocr = OCR('sans-serif', line_height=26.7, space_coefficient=0.6)
# string = ocr.image_to_text('sans-serif-image-rotated.png', rotated=True)
# print(string)