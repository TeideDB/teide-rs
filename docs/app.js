/* Teide Documentation - Sidebar & Highlighting */
(function () {
  'use strict';

  /* --- Mobile sidebar toggle --- */
  var hamburger = document.querySelector('.hamburger');
  var sidebar = document.querySelector('.sidebar');
  var overlay = document.querySelector('.sidebar-overlay');

  function openSidebar() {
    sidebar.classList.add('open');
    overlay.classList.add('open');
  }
  function closeSidebar() {
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
  }
  if (hamburger) hamburger.addEventListener('click', openSidebar);
  if (overlay) overlay.addEventListener('click', closeSidebar);

  /* --- Active link highlighting --- */
  var path = location.pathname;
  // Normalise: strip trailing index.html
  path = path.replace(/index\.html$/, '');
  var links = document.querySelectorAll('.sidebar-link');
  links.forEach(function (a) {
    var href = a.getAttribute('href');
    if (!href) return;
    // Resolve relative href to absolute for comparison
    var resolved = new URL(href, a.baseURI).pathname.replace(/index\.html$/, '');
    if (resolved === path) a.classList.add('active');
  });

  /* --- Lightweight SQL syntax highlighting --- */
  var SQL_KEYWORDS = [
    'SELECT','FROM','WHERE','INSERT','INTO','VALUES','CREATE','TABLE','DROP',
    'AS','ON','JOIN','INNER','LEFT','RIGHT','FULL','OUTER','CROSS',
    'GROUP','BY','ORDER','HAVING','LIMIT','OFFSET','DISTINCT','ALL',
    'UNION','INTERSECT','EXCEPT','AND','OR','NOT','IN','BETWEEN',
    'LIKE','ILIKE','IS','NULL','CASE','WHEN','THEN','ELSE','END',
    'CAST','FILTER','OVER','PARTITION','ROWS','RANGE','UNBOUNDED',
    'PRECEDING','FOLLOWING','CURRENT','ROW','WITH','IF','EXISTS',
    'REPLACE','ASC','DESC','NULLS','FIRST','LAST','SET','TRUE','FALSE',
    'RECURSIVE'
  ];
  var SQL_FUNCTIONS = [
    'COUNT','SUM','AVG','MIN','MAX','FIRST_VALUE','LAST_VALUE','NTH_VALUE',
    'ROW_NUMBER','RANK','DENSE_RANK','NTILE','LAG','LEAD',
    'ABS','CEIL','CEILING','FLOOR','SQRT','ROUND','LN','LOG','EXP',
    'LEAST','GREATEST','UPPER','LOWER','LENGTH','LEN','CHAR_LENGTH',
    'TRIM','BTRIM','SUBSTR','SUBSTRING','CONCAT','COALESCE','NULLIF',
    'EXTRACT','DATE_TRUNC','DATE_DIFF','DATEDIFF','NOW',
    'CURRENT_DATE','CURRENT_TIMESTAMP',
    'STDDEV','STDDEV_SAMP','STDDEV_POP','VARIANCE','VAR_SAMP','VAR_POP',
    'COUNT_DISTINCT','CHARACTER_LENGTH',
    'READ_CSV','READ_PARTED','READ_SPLAYED'
  ];
  var SQL_TYPES = [
    'BOOLEAN','BOOL','INTEGER','INT','INT4','BIGINT','INT8','INT64',
    'REAL','DOUBLE','FLOAT','VARCHAR','TEXT','STRING',
    'DATE','TIME','TIMESTAMP','SYM','SMALLINT','NUMERIC','DECIMAL',
    'DOUBLE PRECISION','CHAR'
  ];

  var kwRe = new RegExp('\\b(' + SQL_KEYWORDS.join('|') + ')\\b', 'g');
  var fnRe = new RegExp('\\b(' + SQL_FUNCTIONS.join('|') + ')\\s*(?=\\()', 'gi');
  var tyRe = new RegExp('\\b(' + SQL_TYPES.join('|') + ')\\b', 'gi');
  var strRe = /'[^']*'/g;
  var numRe = /\b\d+(\.\d+)?\b/g;
  var cmRe = /--[^\n]*/g;

  function highlightSQL(text) {
    // Protect strings and comments first
    var tokens = [];
    var i = 0;
    function protect(re, cls) {
      text = text.replace(re, function (m) {
        var id = '\x00' + i + '\x00';
        tokens.push({ id: id, html: '<span class="' + cls + '">' + esc(m) + '</span>' });
        i++;
        return id;
      });
    }
    function esc(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    protect(cmRe, 'cm');
    protect(strRe, 'str');

    // Escape remaining HTML
    text = esc(text);

    // Highlight types before keywords (some overlap)
    text = text.replace(tyRe, '<span class="ty">$1</span>');
    text = text.replace(fnRe, function (m, fn) {
      return '<span class="fn">' + fn + '</span>';
    });
    text = text.replace(kwRe, '<span class="kw">$1</span>');
    text = text.replace(numRe, '<span class="num">$&</span>');

    // Restore protected tokens
    tokens.forEach(function (t) {
      text = text.replace(t.id, t.html);
    });
    return text;
  }

  document.querySelectorAll('pre code.language-sql').forEach(function (block) {
    block.innerHTML = highlightSQL(block.textContent);
  });
})();
