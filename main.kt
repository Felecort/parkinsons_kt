package smile

import smile.classification.GradientTreeBoost
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat

fun main() {
    val dataset = Read.csv("src/main/resources/parkinsons_updrs.csv", CSVFormat.DEFAULT.withFirstRecordAsHeader())
    val formula = Formula.lhs("total_UPDRS")

    val res = CrossValidation.regression(
        6, formula, dataset,
        { f, data -> GradientTreeBoost.fit(f, data) }
    )

    println(res)
}
