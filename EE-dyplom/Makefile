xelatex : clean
	xelatex EE-dyplom
	biber EE-dyplom
# 	makeglossaries EE-dyplom
	xelatex EE-dyplom
	xelatex EE-dyplom

pdflatex : clean
	pdflatex EE-dyplom
	biber EE-dyplom
# 	makeglossaries EE-dyplom
	pdflatex EE-dyplom
	pdflatex EE-dyplom
	
# # it replaces also spaces in mathematical formulas
# # and possibly in other not intended places
# # But probably not possible with regex to replace correctly
# sololetters:
# 	sed -E 's/(^|[^[:alpha:]\\])([wAIzuoWaiZUO])[[:space:]]+/\1\2~/g' tekst/*.tex

clean :
	-rm *.acn *.app *.aux *.bbl *.blg *.bcf *.glo *.ist *.lof *.log *.lot *.out *.pdf *.run.xml *.slo *.toc
