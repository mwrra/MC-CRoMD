# -*- coding: utf-8 -*-
"""الجزء 11: إنشاء التقارير النهائية بعد اكتمال الأجزاء 1-10."""
from Pre6_Core_TimeLimited import main

if __name__ == "__main__":
    main(
        default_stage="Reports",
        default_models=("SVM_RBF",),
        default_part_name="Pre6_Part11_Final_Reports",
        default_max_hours=0.0,
    )
