<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>

<xsl:template match="/faq">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>
<html>

<xsl:variable name="swidk">
<xsl:choose>
  <xsl:when test="$dir=4">../../../..</xsl:when>
  <xsl:when test="$dir=3">../../..</xsl:when>
  <xsl:when test="$dir=2">../..</xsl:when>
  <xsl:when test="$dir=1">..</xsl:when>
</xsl:choose>
</xsl:variable>

<head>
<title><xsl:value-of select="@name" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
<xsl:value-of select="concat($swidk,'/doc/doc_styles.css')" />
</xsl:attribute>
</link>

</head>
<body>

<!-- *************************************************************** -->
<!-- *************** STANDARD SCI RESEARCH HEADER ****************** -->
<!-- *************************************************************** -->

<center>
<img usemap="#head-links" height="71" width="600" border="0">
<xsl:attribute name="src">
<xsl:value-of select="concat($swidk,'/doc/images/research_menuheader.gif')" />
</xsl:attribute>
</img>
</center>
<map name="head-links">
        <area shape="rect" coords="491,15,567,32" href="http://www.sci.utah.edu/research/research.html"/>
        <area shape="rect" coords="31,9,95,36" href="http://www.sci.utah.edu"/>
 
        <area coords="0,45,150,70" shape="rect" >
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/InstallGuide/installguide.html')" />
        </xsl:attribute>
        </area>

 
        <area coords="150,45,300,70" shape="rect" > 
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/UserGuide/userguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="300,45,450,70" shape="rect" >
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/DeveloperGuide/devguide.html')" />
        </xsl:attribute>
        </area>
 
        <area coords="450,45,600,70" shape="rect" >  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/ReferenceGuide/refguide.html')" />
        </xsl:attribute>
        </area>
</map> 

<!-- *************************************************************** -->
<!-- *************************************************************** -->


<p class="title">
<xsl:value-of select="@name" />
</p>

<xsl:for-each select="/faq/entry">
  <xsl:variable name="num"><xsl:number/></xsl:variable>
  <xsl:for-each select="./question">
    <xsl:for-each select="./p">
      <p class="firstclass"><b><i><xsl:value-of select="."/></i></b></p>
    </xsl:for-each>
    <xsl:for-each select="./pre">
      <pre class="example"><b><i><xsl:value-of select="."/></i></b></pre>
    </xsl:for-each>
    <xsl:for-each select="./img">
      <img><xsl:value-of select="@src"/></img>
    </xsl:for-each>
  </xsl:for-each>
  <xsl:for-each select="./answer">
    <xsl:for-each select="./p">
      <p class="firstclass"><xsl:value-of select="."/></p>
    </xsl:for-each>
    <xsl:for-each select="./pre">
      <pre class="example"><xsl:value-of select="."/></pre>
    </xsl:for-each>
    <xsl:for-each select="./img">
      <p class="firstpara"><center><img>
        <xsl:attribute name="src">
        <xsl:value-of select="."/>
        </xsl:attribute>
      </img></center></p>
    </xsl:for-each>
  </xsl:for-each>
</xsl:for-each>

<!--

<xsl:for-each select="/faq/entry">
  <p class="firstpara"><xsl:value-of select="./question" /></p>
  <p class="firstpara"><xsl:value-of select="./answer" /></p> 
</xsl:for-each>

-->

<!-- ******************************************************************* -->
<!-- *********************** STANDARD SCI FOOTER *********************** -->
<!-- ******************************************************************* -->
<center>
<hr size="1" width="600" />
<font size="-1"><a href="http://www.sci.utah.edu">Scientific Computing and Imaging Institute</a> &#149; <a href="http://www.utah.edu">University of Utah</a> &#149; 
(801) 585-1867</font>
</center>
<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>
</xsl:stylesheet>
