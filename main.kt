package smile

import smile.regression.RandomForest
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat


fun main() {
    val dataset = Read.csv("./parkinsons_updrs.data", CSVFormat.DEFAULT.withFirstRecordAsHeader())
    val formula = Formula.lhs("status")
    
    val res = CrossValidation.regression(
        6, formula, dataset,
        { f, data -> RandomForest.fit(f, data) })

    println(res)
}