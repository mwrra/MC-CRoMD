# -*- coding: utf-8 -*-
"""الجزء 10 من Pre6. أعد تشغيل الملف نفسه إذا توقف بعد 9 ساعات."""
from Pre6_Core_TimeLimited import main

if __name__ == "__main__":
    main(
        default_stage='Tuning',
        default_models=('SVM_Poly',),
        default_part_name='Pre6_Part10_Tuning_SVM_Poly',
        default_max_hours=9.0,
    )
