#According to: https://nbgrader.readthedocs.io/en/stable/user_guide/managing_assignment_files.html#setting-up-the-exchange
# Template for the nbgrader configuration file
# You must change the data to fit your system
c = get_config()
c.CourseDirectory.course_id = "ADSP_seminars_ws22" # Don't change this!!!
c.Exchange.root = "/home/username/WS22_Seminars/instructor/nextcloud" # Change this to point to your local folder (nextcloud).
c.CourseDirectory.student_id = "6666927" # Change this to your matrikel number.

