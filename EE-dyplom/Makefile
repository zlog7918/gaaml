xelatex : clean
	xelatex EE-dyplom
	biber EE-dyplom
	makeglossaries EE-dyplom
	xelatex EE-dyplom
	xelatex EE-dyplom

pdflatex : clean
	pdflatex EE-dyplom
	biber EE-dyplom
	makeglossaries EE-dyplom
	pdflatex EE-dyplom
	pdflatex EE-dyplom
	
# # It replaces also spaces in mathematical formulas
# # and possibly in other not intended places.
# # But (probably) not possible
# # with only regex to replace correctly.
# # Inputs for vscode search and replace option:
# #	search: (^|[^a-zA-Z0-9_ĄąĆćĘęŁłŃńÓóŚśŹźŻż\\])([wAIzuoWaiZUO])[ \t\n]+
# # replace: $1$2~
# sololetters:
# 	sed -E 's/(^|[^[:alpha:]\\])([wAIzuoWaiZUO])[[:space:]]+/\1\2~/g' tekst/*.tex

clean :
	-rm *.acn *.app *.aux *.bbl *.blg *.bcf *.glo *.ist *.lof *.log *.lot *.out *.pdf *.run.xml *.slo *.toc
