# -*- coding: utf-8 -*-
"""الجزء 06 من Pre6. أعد تشغيل الملف نفسه إذا توقف بعد 9 ساعات."""
from Pre6_Core_TimeLimited import main

if __name__ == "__main__":
    main(
        default_stage='Tuning',
        default_models=('KNN', 'DecisionTree', 'LogisticRegression', 'SGD', 'GaussianNB'),
        default_part_name='Pre6_Part06_Tuning_FastModels',
        default_max_hours=9.0,
    )
