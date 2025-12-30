'''drawn from Gloria github: https://github.com/marshuang80/gloria
'''

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

LC25000_lung = {
    "Lung adenocarcinoma": [
        "Histopathology slide of Lung adenocarcinoma with glandular patterns",
        "Histopathology image showing Lung adenocarcinoma cells with mucin production",
        "Pathology tissue sample of Lung adenocarcinoma with lepidic growth pattern"
    ],

    "Benign lung": [
        "Histopathology slide of Benign lung tissue with normal alveolar architecture",
        "Histopathology image showing Benign lung tissue with no abnormal cell growth",
        "Pathology tissue sample of Benign lung with fibrous tissue and no signs of malignancy"
    ],

    "Lung squamous cell carcinoma": [
        "Histopathology slide of Lung squamous cell carcinoma with keratin pearls",
        "Histopathology image showing Lung squamous cell carcinoma cells with intercellular bridges",
        "Pathology tissue sample of Lung squamous cell carcinoma with squamous differentiation"
    ]
}

LC25000_colon = {
    "Colon adenocarcinoma": [
        "Histopathology slide of Colon adenocarcinoma with glandular structures",
        "Histopathology image showing Colon adenocarcinoma cells with mucinous differentiation",
        "Pathology tissue sample of Colon adenocarcinoma with invasion into surrounding tissue"
    ],

    "Benign colonic tissue": [
        "Histopathology slide of Benign colonic tissue with normal mucosal architecture",
        "Histopathology image showing Benign colonic tissue with no dysplastic changes",
        "Pathology tissue sample of Benign colonic tissue without malignant features"
    ]
}