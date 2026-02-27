/* Teide Documentation - Sidebar, Active Links & Syntax Highlighting */
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
  path = path.replace(/index\.html$/, '');
  var links = document.querySelectorAll('.sidebar-link');
  links.forEach(function (a) {
    var href = a.getAttribute('href');
    if (!href) return;
    var resolved = new URL(href, a.baseURI).pathname.replace(/index\.html$/, '');
    if (resolved === path) a.classList.add('active');
  });

  /* === Shared helpers === */
  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function makeHighlighter(protectors, replacers) {
    return function (text) {
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
      protectors.forEach(function (p) { protect(p.re, p.cls); });
      text = esc(text);
      replacers.forEach(function (r) {
        if (r.fn) {
          text = text.replace(r.re, r.fn);
        } else {
          text = text.replace(r.re, r.tpl);
        }
      });
      tokens.forEach(function (t) { text = text.replace(t.id, t.html); });
      return text;
    };
  }

  /* === SQL Highlighter === */
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

  var highlightSQL = makeHighlighter(
    [
      { re: /--[^\n]*/g, cls: 'cm' },
      { re: /'[^']*'/g, cls: 'str' }
    ],
    [
      { re: new RegExp('\\b(' + SQL_TYPES.join('|') + ')\\b', 'gi'), tpl: '<span class="ty">$1</span>' },
      { re: new RegExp('\\b(' + SQL_FUNCTIONS.join('|') + ')\\s*(?=\\()', 'gi'), fn: function (m, f) { return '<span class="fn">' + f + '</span>'; } },
      { re: new RegExp('\\b(' + SQL_KEYWORDS.join('|') + ')\\b', 'g'), tpl: '<span class="kw">$1</span>' },
      { re: /\b\d+(\.\d+)?\b/g, tpl: '<span class="num">$&</span>' }
    ]
  );

  /* === Rust Highlighter === */
  var RUST_KEYWORDS = [
    'as','async','await','break','const','continue','crate','dyn','else','enum',
    'extern','false','fn','for','if','impl','in','let','loop','match','mod',
    'move','mut','pub','ref','return','self','Self','static','struct','super',
    'trait','true','type','unsafe','use','where','while','Ok','Err','Some','None'
  ];
  var RUST_TYPES = [
    'bool','char','f32','f64','i8','i16','i32','i64','i128','isize',
    'u8','u16','u32','u64','u128','usize','str','String','Vec','Box',
    'Option','Result','Arc','Mutex','Rc','HashMap','PathBuf','Path',
    'Context','Table','Graph','Column','Session','ExecResult','SqlResult',
    'AggOp','WindowFunc','FrameType','FrameBound','MemStats','Error'
  ];
  var RUST_MACROS = [
    'println','eprintln','print','format','vec','panic','assert','assert_eq',
    'assert_ne','cfg','derive','todo','unimplemented','unreachable','writeln','write'
  ];

  var highlightRust = makeHighlighter(
    [
      { re: /\/\/[^\n]*/g, cls: 'cm' },
      { re: /"(?:[^"\\]|\\.)*"/g, cls: 'str' }
    ],
    [
      { re: new RegExp('\\b(' + RUST_MACROS.join('|') + ')!', 'g'), tpl: '<span class="fn">$1</span>!' },
      { re: new RegExp('\\b(' + RUST_TYPES.join('|') + ')\\b', 'g'), tpl: '<span class="ty">$1</span>' },
      { re: new RegExp('\\b(' + RUST_KEYWORDS.join('|') + ')\\b', 'g'), tpl: '<span class="kw">$1</span>' },
      { re: /\b\d+(\.\d+)?\b/g, tpl: '<span class="num">$&</span>' },
      { re: /&amp;(mut\b|str\b|self\b|\[)/g, tpl: '<span class="op">&amp;</span>$1' }
    ]
  );

  /* === Bash Highlighter === */
  var BASH_KEYWORDS = [
    'if','then','else','elif','fi','for','while','do','done','case','esac',
    'in','function','return','exit','export','source','alias','cd','echo',
    'sudo','chmod','chown','mkdir','rm','cp','mv','ls','cat','grep','find',
    'cargo','git','psql','teide','teide-server'
  ];

  var highlightBash = makeHighlighter(
    [
      { re: /#[^\n]*/g, cls: 'cm' },
      { re: /"(?:[^"\\]|\\.)*"/g, cls: 'str' },
      { re: /'[^']*'/g, cls: 'str' }
    ],
    [
      { re: /--[\w][\w-]*/g, tpl: '<span class="op">$&</span>' },
      { re: /-[a-zA-Z]\b/g, tpl: '<span class="op">$&</span>' },
      { re: new RegExp('\\b(' + BASH_KEYWORDS.join('|') + ')\\b', 'g'), tpl: '<span class="kw">$1</span>' },
      { re: /\$[\w]+/g, tpl: '<span class="fn">$&</span>' },
      { re: /\b\d+\b/g, tpl: '<span class="num">$&</span>' }
    ]
  );

  /* === TOML / Config Highlighter === */
  var highlightToml = makeHighlighter(
    [
      { re: /#[^\n]*/g, cls: 'cm' },
      { re: /"(?:[^"\\]|\\.)*"/g, cls: 'str' }
    ],
    [
      { re: /^\[([^\]]+)\]/gm, tpl: '<span class="kw">[$1]</span>' },
      { re: /^(\w[\w-]*)\s*=/gm, fn: function (m, k) { return '<span class="fn">' + k + '</span> ='; } },
      { re: /\b(true|false)\b/g, tpl: '<span class="kw">$1</span>' },
      { re: /\b\d+(\.\d+)?\b/g, tpl: '<span class="num">$&</span>' }
    ]
  );

  /* === Apply highlighting === */
  var highlighters = {
    'language-sql': highlightSQL,
    'language-rust': highlightRust,
    'language-rs': highlightRust,
    'language-bash': highlightBash,
    'language-sh': highlightBash,
    'language-shell': highlightBash,
    'language-toml': highlightToml
  };

  document.querySelectorAll('pre code').forEach(function (block) {
    var cls = block.className;
    for (var lang in highlighters) {
      if (cls.indexOf(lang) !== -1) {
        block.innerHTML = highlighters[lang](block.textContent);
        return;
      }
    }
  });
})();
