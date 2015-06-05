man3dir = $(mandir)/man3
docdir  = $(prefix)/doc/@PACKAGE@-@VERSION@

all-local: dox #pdf

dox: html/index.html
html/index.html: Doxyfile
	@DOXYGEN@

pdf: @PACKAGE@.pdf
@PACKAGE@.pdf:
	$(MAKE) -C ./latex pdf
	ln -s ./latex/refman.ps	 @PACKAGE@.ps
	ln -s ./latex/refman.pdf @PACKAGE@.pdf

install-data-local:
	@mkdir -p $(srcdir)/html # workaround to allow user to install without doc
	$(mkinstalldirs) $(DESTDIR)$(docdir)
	@cp -rp $(srcdir)/html $(docdir)

clean-local:
	$(RM) -r latex
	$(RM) -r html man @PACKAGE@.ps @PACKAGE@.pdf
