# Leafy Plant Disease Dataset

Leafy is a clean train/validation/test image dataset for multi-class plant leaf disease classification. The dataset is already split and ready for PyTorch `ImageFolder` training.

## Dataset Version

| Field | Value |
|---|---:|
| Classes | 90 |
| Total images | 137,186 |
| Train images | 109,736 |
| Validation images | 13,759 |
| Test images | 13,691 |
| Naming standard | `Species___condition` |
| Cross-split exact duplicate hash groups | 0 |

## Intended Use

Use this dataset for supervised plant disease image classification, transfer learning, and benchmarking. Each image belongs to exactly one crop-and-condition class. The dataset is useful for model development and research, but model predictions should not replace expert field diagnosis without additional validation.

## Directory Structure

```text
data_split/
  train/
    Species___condition/
      image files
  val/
    Species___condition/
      image files
  test/
    Species___condition/
      image files
  split_summary.json
  dataset_fingerprint.json
```

## Plant Coverage

| Plant | Classes | Images |
|---|---:|---:|
| Apple | 7 | 6,542 |
| Blueberry | 1 | 1,502 |
| Cassava | 5 | 21,397 |
| Cherry | 2 | 1,906 |
| Coffee | 2 | 1,393 |
| Corn | 4 | 4,433 |
| Cucumber | 2 | 691 |
| Grape | 4 | 7,432 |
| Guava | 1 | 277 |
| Jamun | 2 | 624 |
| Mango | 1 | 265 |
| Orange | 1 | 5,507 |
| Peach | 2 | 2,657 |
| Pepper bell | 2 | 2,475 |
| Pomegranate | 2 | 559 |
| Potato | 8 | 9,710 |
| Raspberry | 1 | 371 |
| Rice | 8 | 9,966 |
| Rose | 3 | 14,910 |
| Soybean | 3 | 11,512 |
| Squash | 1 | 1,835 |
| Strawberry | 2 | 1,565 |
| Sugarcane | 5 | 2,521 |
| Tea | 5 | 1,490 |
| Tomato | 10 | 21,374 |
| Watermelon | 3 | 1,000 |
| Wheat | 3 | 3,272 |

## Class Counts and Descriptions

| Class | Plant | Condition | Train | Val | Test | Total | Description |
|---|---|---|---:|---:|---:|---:|---|
| `Apple___alternaria_leaf_spot` | Apple | alternaria leaf spot | 222 | 28 | 28 | 278 | Alternaria leaf spot symptoms with dark lesions and chlorotic halos. |
| `Apple___black_rot` | Apple | black rot | 497 | 62 | 62 | 621 | black rot symptoms with dark necrotic lesions and vein-associated damage. |
| `Apple___brown_spot` | Apple | brown spot | 171 | 21 | 23 | 215 | brown spot disease symptoms with tan to brown lesions. |
| `Apple___gray_spot` | Apple | gray spot | 317 | 39 | 39 | 395 | gray spot disease symptoms with grayish necrotic spotting. |
| `Apple___healthy` | Apple | healthy | 2,054 | 258 | 258 | 2,570 | healthy leaves without visible disease symptoms. |
| `Apple___rust` | Apple | rust | 993 | 124 | 124 | 1,241 | rust disease symptoms with orange, yellow, or brown pustules or lesions. |
| `Apple___scab` | Apple | scab | 978 | 122 | 122 | 1,222 | scab disease symptoms with rough, dark, or corky lesions. |
| `Blueberry___healthy` | Blueberry | healthy | 1,202 | 150 | 150 | 1,502 | healthy leaves without visible disease symptoms. |
| `Cassava___bacterial_blight` | Cassava | bacterial blight | 870 | 109 | 108 | 1,087 | bacterial blight symptoms such as water-soaked lesions, browning, or tissue necrosis. |
| `Cassava___brown_streak_disease` | Cassava | brown streak disease | 1,751 | 219 | 219 | 2,189 | brown streak disease symptoms including streaking, chlorosis, and necrotic damage. |
| `Cassava___green_mottle` | Cassava | green mottle | 1,909 | 239 | 238 | 2,386 | green mottle symptoms with mottled green chlorotic patterns. |
| `Cassava___healthy` | Cassava | healthy | 2,062 | 258 | 257 | 2,577 | healthy leaves without visible disease symptoms. |
| `Cassava___mosaic_disease` | Cassava | mosaic disease | 10,526 | 1,316 | 1,316 | 13,158 | mosaic disease symptoms with mottled yellow-green leaf patterns. |
| `Cherry___healthy` | Cherry | healthy | 683 | 85 | 86 | 854 | healthy leaves without visible disease symptoms. |
| `Cherry___powdery_mildew` | Cherry | powdery mildew | 842 | 105 | 105 | 1,052 | powdery mildew symptoms with pale or white fungal growth on leaves. |
| `Coffee___healthy` | Coffee | healthy | 632 | 80 | 79 | 791 | healthy leaves without visible disease symptoms. |
| `Coffee___rust` | Coffee | rust | 481 | 62 | 59 | 602 | rust disease symptoms with orange, yellow, or brown pustules or lesions. |
| `Corn___common_rust` | Corn | common rust | 954 | 119 | 119 | 1,192 | common rust symptoms visible on leaf images. |
| `Corn___gray_leaf_spot` | Corn | gray leaf spot | 878 | 109 | 107 | 1,094 | gray leaf spot lesions, typically rectangular or gray-brown on leaves. |
| `Corn___healthy` | Corn | healthy | 930 | 116 | 116 | 1,162 | healthy leaves without visible disease symptoms. |
| `Corn___northern_leaf_blight` | Corn | northern leaf blight | 788 | 98 | 99 | 985 | northern leaf blight symptoms visible on leaf images. |
| `Cucumber___diseased` | Cucumber | diseased | 277 | 38 | 35 | 350 | general diseased leaf examples where a more specific disease label was not available. |
| `Cucumber___healthy` | Cucumber | healthy | 273 | 34 | 34 | 341 | healthy leaves without visible disease symptoms. |
| `Grape___black_measles` | Grape | black measles | 2,140 | 269 | 262 | 2,671 | grape black measles symptoms with dark spotting and leaf discoloration. |
| `Grape___black_rot` | Grape | black rot | 1,262 | 160 | 158 | 1,580 | black rot symptoms with dark necrotic lesions and vein-associated damage. |
| `Grape___healthy` | Grape | healthy | 1,371 | 172 | 162 | 1,705 | healthy leaves without visible disease symptoms. |
| `Grape___leaf_blight` | Grape | leaf blight | 1,176 | 157 | 143 | 1,476 | leaf blight symptoms with broad necrotic or scorched regions. |
| `Guava___healthy` | Guava | healthy | 222 | 28 | 27 | 277 | healthy leaves without visible disease symptoms. |
| `Jamun___diseased` | Jamun | diseased | 276 | 34 | 35 | 345 | general diseased leaf examples where a more specific disease label was not available. |
| `Jamun___healthy` | Jamun | healthy | 223 | 28 | 28 | 279 | healthy leaves without visible disease symptoms. |
| `Mango___diseased` | Mango | diseased | 212 | 26 | 27 | 265 | general diseased leaf examples where a more specific disease label was not available. |
| `Orange___citrus_greening` | Orange | citrus greening | 4,406 | 551 | 550 | 5,507 | citrus greening symptoms with blotchy mottling and yellowing. |
| `Peach___bacterial_spot` | Peach | bacterial spot | 1,838 | 230 | 229 | 2,297 | bacterial spot lesions and speckling on leaf surfaces. |
| `Peach___healthy` | Peach | healthy | 288 | 36 | 36 | 360 | healthy leaves without visible disease symptoms. |
| `Pepper_bell___bacterial_spot` | Pepper bell | bacterial spot | 798 | 100 | 99 | 997 | bacterial spot lesions and speckling on leaf surfaces. |
| `Pepper_bell___healthy` | Pepper bell | healthy | 1,182 | 148 | 148 | 1,478 | healthy leaves without visible disease symptoms. |
| `Pomegranate___diseased` | Pomegranate | diseased | 218 | 27 | 27 | 272 | general diseased leaf examples where a more specific disease label was not available. |
| `Pomegranate___healthy` | Pomegranate | healthy | 230 | 29 | 28 | 287 | healthy leaves without visible disease symptoms. |
| `Potato___bacterial_wilt` | Potato | bacterial wilt | 455 | 57 | 57 | 569 | bacterial wilt symptoms visible on leaf images. |
| `Potato___early_blight` | Potato | early blight | 2,103 | 263 | 262 | 2,628 | early blight leaf lesions, often with dark concentric spotting. |
| `Potato___healthy` | Potato | healthy | 1,820 | 227 | 228 | 2,275 | healthy leaves without visible disease symptoms. |
| `Potato___late_blight` | Potato | late blight | 1,671 | 208 | 208 | 2,087 | late blight symptoms including irregular dark lesions and leaf collapse. |
| `Potato___leafroll_virus` | Potato | leafroll virus | 422 | 53 | 52 | 527 | potato leafroll virus symptoms with curled, stiff, or yellowing leaves. |
| `Potato___mosaic_virus` | Potato | mosaic virus | 533 | 66 | 67 | 666 | viral mosaic symptoms with mottling, chlorosis, and uneven leaf coloration. |
| `Potato___pests` | Potato | pests | 488 | 62 | 61 | 611 | visible pest-related potato leaf damage. |
| `Potato___phytophthora` | Potato | phytophthora | 277 | 34 | 36 | 347 | Phytophthora-related potato disease symptoms. |
| `Raspberry___healthy` | Raspberry | healthy | 297 | 37 | 37 | 371 | healthy leaves without visible disease symptoms. |
| `Rice___bacterial_blight` | Rice | bacterial blight | 1,243 | 169 | 172 | 1,584 | bacterial blight symptoms such as water-soaked lesions, browning, or tissue necrosis. |
| `Rice___blast` | Rice | blast | 1,161 | 139 | 140 | 1,440 | blast disease symptoms with spindle-shaped or necrotic lesions. |
| `Rice___brown_spot` | Rice | brown spot | 1,283 | 155 | 162 | 1,600 | brown spot disease symptoms with tan to brown lesions. |
| `Rice___healthy` | Rice | healthy | 1,190 | 149 | 149 | 1,488 | healthy leaves without visible disease symptoms. |
| `Rice___hispa` | Rice | hispa | 452 | 56 | 57 | 565 | rice hispa pest damage with scraping, streaking, or pale feeding injury. |
| `Rice___leaf_blast` | Rice | leaf blast | 785 | 98 | 98 | 981 | rice leaf blast symptoms visible on leaf blades. |
| `Rice___neck_blast` | Rice | neck blast | 800 | 100 | 100 | 1,000 | rice neck blast symptoms affecting panicles and neck tissue. |
| `Rice___tungro` | Rice | tungro | 1,046 | 131 | 131 | 1,308 | rice tungro viral symptoms including yellow-orange discoloration and stunting. |
| `Rose___healthy` | Rose | healthy | 3,974 | 516 | 488 | 4,978 | healthy leaves without visible disease symptoms. |
| `Rose___rust` | Rose | rust | 3,956 | 501 | 496 | 4,953 | rust disease symptoms with orange, yellow, or brown pustules or lesions. |
| `Rose___slug_sawfly` | Rose | slug sawfly | 3,982 | 500 | 497 | 4,979 | slug sawfly feeding damage on rose leaves. |
| `Soybean___caterpillar` | Soybean | caterpillar | 2,647 | 331 | 331 | 3,309 | caterpillar feeding damage such as chewing, holes, and edge loss. |
| `Soybean___diabrotica_speciosa` | Soybean | diabrotica speciosa | 1,764 | 220 | 221 | 2,205 | Diabrotica speciosa pest damage on soybean leaves. |
| `Soybean___healthy` | Soybean | healthy | 4,803 | 597 | 598 | 5,998 | healthy leaves without visible disease symptoms. |
| `Squash___powdery_mildew` | Squash | powdery mildew | 1,468 | 184 | 183 | 1,835 | powdery mildew symptoms with pale or white fungal growth on leaves. |
| `Strawberry___healthy` | Strawberry | healthy | 365 | 46 | 45 | 456 | healthy leaves without visible disease symptoms. |
| `Strawberry___leaf_scorch` | Strawberry | leaf scorch | 887 | 111 | 111 | 1,109 | leaf scorch symptoms visible on leaf images. |
| `Sugarcane___healthy` | Sugarcane | healthy | 414 | 56 | 52 | 522 | healthy leaves without visible disease symptoms. |
| `Sugarcane___mosaic` | Sugarcane | mosaic | 368 | 42 | 52 | 462 | mosaic symptoms visible on leaf images. |
| `Sugarcane___red_rot` | Sugarcane | red rot | 414 | 52 | 52 | 518 | red rot disease symptoms associated with reddish internal or external tissue damage. |
| `Sugarcane___rust` | Sugarcane | rust | 412 | 53 | 49 | 514 | rust disease symptoms with orange, yellow, or brown pustules or lesions. |
| `Sugarcane___yellow_leaf` | Sugarcane | yellow leaf | 404 | 50 | 51 | 505 | yellow leaf symptoms with chlorosis and leaf discoloration. |
| `Tea___algal_leaf` | Tea | algal leaf | 271 | 34 | 34 | 339 | algal leaf spot symptoms with raised or discolored patches. |
| `Tea___anthracnose` | Tea | anthracnose | 240 | 30 | 30 | 300 | anthracnose symptoms with dark sunken lesions and necrosis. |
| `Tea___bird_eye_spot` | Tea | bird eye spot | 240 | 31 | 29 | 300 | bird eye spot symptoms with small round lesions and pale centers. |
| `Tea___brown_blight` | Tea | brown blight | 271 | 34 | 34 | 339 | brown blight symptoms with brown necrotic patches. |
| `Tea___healthy` | Tea | healthy | 170 | 21 | 21 | 212 | healthy leaves without visible disease symptoms. |
| `Tomato___bacterial_spot` | Tomato | bacterial spot | 1,702 | 213 | 212 | 2,127 | bacterial spot lesions and speckling on leaf surfaces. |
| `Tomato___early_blight` | Tomato | early blight | 800 | 100 | 100 | 1,000 | early blight leaf lesions, often with dark concentric spotting. |
| `Tomato___healthy` | Tomato | healthy | 1,273 | 159 | 159 | 1,591 | healthy leaves without visible disease symptoms. |
| `Tomato___late_blight` | Tomato | late blight | 1,527 | 191 | 191 | 1,909 | late blight symptoms including irregular dark lesions and leaf collapse. |
| `Tomato___leaf_curl` | Tomato | leaf curl | 6,866 | 854 | 851 | 8,571 | leaf curl symptoms commonly associated with viral stress and distorted leaf growth. |
| `Tomato___leaf_mold` | Tomato | leaf mold | 762 | 95 | 95 | 952 | leaf mold symptoms including pale patches and mold-like growth. |
| `Tomato___mosaic_virus` | Tomato | mosaic virus | 298 | 37 | 38 | 373 | viral mosaic symptoms with mottling, chlorosis, and uneven leaf coloration. |
| `Tomato___septoria_leaf_spot` | Tomato | septoria leaf spot | 1,417 | 177 | 177 | 1,771 | Septoria leaf spot symptoms with small circular lesions and spotting. |
| `Tomato___spider_mites` | Tomato | spider mites | 1,341 | 168 | 167 | 1,676 | spider mite damage symptoms including stippling, bronzing, and stressed leaf tissue. |
| `Tomato___target_spot` | Tomato | target spot | 1,123 | 140 | 141 | 1,404 | target spot symptoms with circular lesions and concentric ring patterns. |
| `Watermelon___downy_mildew` | Watermelon | downy mildew | 304 | 38 | 38 | 380 | downy mildew symptoms visible on leaf images. |
| `Watermelon___healthy` | Watermelon | healthy | 164 | 20 | 21 | 205 | healthy leaves without visible disease symptoms. |
| `Watermelon___mosaic_virus` | Watermelon | mosaic virus | 332 | 42 | 41 | 415 | viral mosaic symptoms with mottling, chlorosis, and uneven leaf coloration. |
| `Wheat___brown_rust` | Wheat | brown rust | 732 | 91 | 92 | 915 | brown rust symptoms with brown to orange pustules on leaves. |
| `Wheat___healthy` | Wheat | healthy | 981 | 122 | 122 | 1,225 | healthy leaves without visible disease symptoms. |
| `Wheat___yellow_rust` | Wheat | yellow rust | 906 | 113 | 113 | 1,132 | yellow rust symptoms with yellow-orange stripe-like pustules. |

## Class Support Notes

The dataset is usable as-is, but it remains naturally imbalanced. Report macro F1 and balanced accuracy in addition to overall accuracy. Classes with smaller validation or test counts should be interpreted with more caution than high-count classes.

## Validation Commands

```bash
python data_preparation/validate_split_dataset.py --data-dir ./data_split --write-summary
python data_preparation/dataset_audit.py --data-dir ./data_split --top-k 25
python data_preparation/deduplicate_dataset.py --data-dir ./data_split --report-json ./reports/duplicate_report.json
```

## Training Command

```bash
python train_efficientnet.py --data-dir ./data_split --architecture efficientnet_v2_s --selection-metric val_macro_f1 --imbalance-strategy ens_loss
```

## Files to Publish With the Dataset

Keep `split_summary.json` and `dataset_fingerprint.json` with the dataset. They record class counts, split counts, validation status, and the class-count fingerprint used by training outputs.
