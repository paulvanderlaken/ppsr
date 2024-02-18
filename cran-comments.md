## winbuilder test results
There were no ERRORs or WARNINGs. 
There were 3 NOTEs.

N  checking dependencies in R code
   Namespaces in Imports field not imported from:
     'rpart' 'withr'
     All declared Imports should be used.

> These are actually used. So I do not understand these being flagged. 
> Once removed their absence results in failures.

N  checking for detritus in the temp directory
   Found the following files/directories:
     'lastMiKTeXException'
     
> I do not find this file and I do not understand where it comes from


## R CMD check results

0 errors v | 0 warnings v | 0 notes v


## R-hub builder results
There were no ERRORs or WARNINGs. 
There were no NOTEs.
