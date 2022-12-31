import pandas as pd

student_mat = pd.read_csv("student-mat.csv")
student_por = pd.read_csv("student-por.csv")

df = pd.merge(
    student_mat,
    student_por,
    on=[
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "nursery",
        "internet",
    ],
)

print(df)
