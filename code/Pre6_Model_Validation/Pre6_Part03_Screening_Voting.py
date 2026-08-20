# -*- coding: utf-8 -*-
"""الجزء 03 من Pre6. أعد تشغيل الملف نفسه إذا توقف بعد 9 ساعات."""
from Pre6_Core_TimeLimited import main

if __name__ == "__main__":
    main(
        default_stage='Screening',
        default_models=('Voting',),
        default_part_name='Pre6_Part03_Screening_Voting',
        default_max_hours=9.0,
    )
