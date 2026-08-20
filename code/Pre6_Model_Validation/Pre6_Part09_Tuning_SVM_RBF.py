# -*- coding: utf-8 -*-
"""الجزء 09 من Pre6. أعد تشغيل الملف نفسه إذا توقف بعد 9 ساعات."""
from Pre6_Core_TimeLimited import main

if __name__ == "__main__":
    main(
        default_stage='Tuning',
        default_models=('SVM_RBF',),
        default_part_name='Pre6_Part09_Tuning_SVM_RBF',
        default_max_hours=9.0,
    )
