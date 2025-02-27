import smile.data.formula.Formula
import smile.data.`type`.DataTypes
import smile.io.Read
import smile.validation.CrossValidation
import smile.regression.GradientTreeBoost
import org.apache.commons.csv.CSVFormat

fun main() {
    // Чтение датасета
    val dataset = Read.csv("src/main/resources/parkinsons_updrs.csv", CSVFormat.DEFAULT.withFirstRecordAsHeader())

    // Удаление поля "name"
    val datasetWithoutName = dataset.drop("name")

    // Определение формулы для регрессии
    val formula = Formula.lhs("total_UPDRS")

    // Кросс-валидация
    val res = CrossValidation.regression(
        6, formula, datasetWithoutName,
        { f, data -> GradientTreeBoost.fit(f, data) }
    )

    // Вывод результата
    println(res)
}