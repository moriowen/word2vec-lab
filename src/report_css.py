"""Design tokens and component styles for the report page."""

CSS = """
  :root {
    --paper:#EFF2F5; --surface:#FFFFFF; --raised:#F7F9FB;
    --ink:#121620; --muted:#58637A; --faint:#8894A8;
    --line:#D8DEE7; --line-2:#C3CCD9;
    --accent:#0D6E86; --accent-soft:#D6EAF0;
    --warm:#A8500C; --warm-soft:#F6E4D2;
    --good:#196B3C; --bad:#9B2C2C; --good-soft:#DCEFE3;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
    --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --paper:#0C0F15; --surface:#141922; --raised:#1A2029;
    --ink:#E7ECF3; --muted:#97A5B9; --faint:#6C7B91;
    --line:#242C38; --line-2:#333D4C;
    --accent:#3FC6E0; --accent-soft:#10323C;
    --warm:#E3A055; --warm-soft:#3A2612;
    --good:#5FC98A; --bad:#E08585; --good-soft:#123021;
  } }
  :root[data-theme="dark"] {
    --paper:#0C0F15; --surface:#141922; --raised:#1A2029;
    --ink:#E7ECF3; --muted:#97A5B9; --faint:#6C7B91;
    --line:#242C38; --line-2:#333D4C;
    --accent:#3FC6E0; --accent-soft:#10323C;
    --warm:#E3A055; --warm-soft:#3A2612;
    --good:#5FC98A; --bad:#E08585; --good-soft:#123021;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
       font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:980px;margin:0 auto;padding:0 24px 96px}
  .prose{max-width:68ch}
  h1,h2,h3{font-family:var(--serif);font-weight:600;text-wrap:balance;margin:0}
  h1{font-size:clamp(2.1rem,5vw,3.1rem);line-height:1.08;letter-spacing:-.015em}
  h2{font-size:clamp(1.4rem,3vw,1.85rem);line-height:1.2}
  h3{font-size:1.08rem;line-height:1.3}
  p{margin:0}
  a{color:var(--accent)}
  code{font-family:var(--mono);font-size:.88em;background:var(--raised);
       border:1px solid var(--line);border-radius:3px;padding:.08em .34em}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
           text-transform:uppercase;color:var(--faint)}
  header.masthead{border-bottom:1px solid var(--line-2);padding:72px 0 36px;
                  display:flex;flex-direction:column;gap:18px}
  .lede{font-size:1.16rem;color:var(--muted);max-width:60ch}
  .runmeta{display:flex;flex-wrap:wrap;gap:8px 22px;font-family:var(--mono);
           font-size:.74rem;color:var(--faint)}
  .runmeta b{color:var(--muted);font-weight:500}
  section{padding-top:60px;display:flex;flex-direction:column;gap:22px}
  .sechead{display:flex;flex-direction:column;gap:8px}
  nav.toc{display:flex;flex-wrap:wrap;gap:6px 10px;font-family:var(--mono);font-size:.74rem;
          padding:16px 0;border-bottom:1px solid var(--line)}
  nav.toc a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}
  nav.toc a:hover{color:var(--accent);border-bottom-color:var(--accent)}

  .models{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:14px}
  .mcard{background:var(--surface);border:1px solid var(--line);border-radius:6px;
         padding:16px 16px 14px;display:flex;flex-direction:column;gap:10px}
  .mcard.is-focus{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
  .mcard .tag{display:flex;align-items:baseline;gap:8px}
  .mcard .id{font-family:var(--mono);font-size:1.5rem;line-height:1;color:var(--accent)}
  .mcard .nm{font-size:.82rem;color:var(--muted);line-height:1.35}
  .acc{font-family:var(--mono);font-size:1.85rem;line-height:1;font-variant-numeric:tabular-nums}
  .bar{height:5px;background:var(--raised);border-radius:3px;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--accent)}
  .mcard.is-focus .bar>i{background:var(--warm)}
  .mmeta{font-family:var(--mono);font-size:.7rem;color:var(--faint);
         display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}

  .figure{background:var(--surface);border:1px solid var(--line);border-radius:6px;
          padding:22px;overflow-x:auto}
  .figure svg{display:block;min-width:620px;width:100%;height:auto}
  .caption{font-size:.84rem;color:var(--muted);margin-top:12px;max-width:66ch}
  .figgrid{display:grid;grid-template-columns:1fr;gap:18px}

  .tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
             border-radius:6px}
  table{border-collapse:collapse;width:100%;font-size:.86rem;min-width:520px}
  th,td{text-align:left;padding:10px 14px;border-top:1px solid var(--line);
        font-variant-numeric:tabular-nums}
  thead th{border-top:0;background:var(--raised);font-family:var(--mono);
           font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);
           font-weight:500;white-space:nowrap}
  td.n,th.n{text-align:right;font-family:var(--mono);font-size:.82rem}
  td.mono,th.mono{font-family:var(--mono);font-size:.8rem}
  tr.hi td{background:var(--accent-soft)}
  td .win{color:var(--good);font-weight:600}
  td .lose{color:var(--bad)}

  .stats{display:flex;flex-wrap:wrap;gap:30px;padding:16px 0;
         border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
  .stat{display:flex;flex-direction:column;gap:3px}
  .stat .n{font-family:var(--mono);font-size:1.35rem;font-variant-numeric:tabular-nums}
  .stat .l{font-size:.76rem;color:var(--faint)}

  .note{background:var(--raised);border-left:2px solid var(--accent);
        padding:14px 18px;font-size:.92rem;color:var(--muted);border-radius:0 4px 4px 0}
  .note b{color:var(--ink);font-weight:600}
  .note.warm{border-left-color:var(--warm)}

  .nbox{background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}
  .nrow{display:grid;grid-template-columns:150px 1fr;border-top:1px solid var(--line)}
  .nrow:first-child{border-top:0}
  .nq{padding:14px 16px;font-family:var(--mono);font-size:.82rem;background:var(--raised);
      border-right:1px solid var(--line);display:flex;align-items:center}
  .npair{display:flex;flex-direction:column}
  .nline{display:grid;grid-template-columns:26px 1fr;gap:10px;padding:10px 16px;align-items:baseline}
  .nline+.nline{border-top:1px dashed var(--line)}
  .nlabel{font-family:var(--mono);font-size:.68rem;color:var(--faint)}
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{display:inline-block;font-family:var(--mono);font-size:.74rem;padding:2px 8px;
        border-radius:3px;background:var(--raised);border:1px solid var(--line);
        color:var(--muted);white-space:nowrap}
  .chip.only-d{background:var(--warm-soft);border-color:var(--warm);color:var(--ink)}
  .chip.agree{background:var(--good-soft);border-color:var(--good);color:var(--ink)}
  .legend{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;font-size:.82rem;
          color:var(--muted);background:var(--raised);border:1px solid var(--line);
          border-radius:6px;padding:10px 14px}
  .legend .chip{flex:none}
  .legend .li{display:flex;align-items:center;gap:8px}

  .driftgrid{display:grid;grid-template-columns:1fr 1fr;gap:28px}
  @media(max-width:720px){.driftgrid{grid-template-columns:1fr}.nrow{grid-template-columns:1fr}
    .nq{border-right:0;border-bottom:1px solid var(--line)}}
  .dcol{display:flex;flex-direction:column;gap:12px}
  .dcol h3{font-family:var(--sans);font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;
           color:var(--faint);font-weight:600}
  .drow{display:grid;grid-template-columns:112px 1fr 48px;align-items:center;gap:10px;font-size:.85rem}
  .drow .w{font-family:var(--mono);font-size:.8rem;overflow:hidden;text-overflow:ellipsis}
  .drow .t{height:9px;background:var(--raised);border-radius:2px;overflow:hidden}
  .drow .t>i{display:block;height:100%}
  .moved .t>i{background:var(--warm)}
  .anchored .t>i{background:var(--line-2)}
  .drow .v{font-family:var(--mono);font-size:.74rem;color:var(--muted);text-align:right;
           font-variant-numeric:tabular-nums}

  ol.steps{list-style:none;margin:0;padding:0;counter-reset:s;display:flex;flex-direction:column}
  ol.steps>li{counter-increment:s;display:grid;grid-template-columns:40px 1fr;gap:18px;
              padding:18px 0;border-top:1px solid var(--line)}
  ol.steps>li:last-child{border-bottom:1px solid var(--line)}
  ol.steps>li::before{content:counter(s,decimal-leading-zero);font-family:var(--mono);
                      font-size:.78rem;color:var(--accent);padding-top:3px}
  .step-b{display:flex;flex-direction:column;gap:7px}
  .step-b .why{font-size:.9rem;color:var(--muted)}

  .log{display:flex;flex-direction:column}
  .lrow{display:grid;grid-template-columns:96px 1fr auto;gap:16px;padding:14px 0;
        border-top:1px solid var(--line);align-items:baseline}
  .lrow:last-child{border-bottom:1px solid var(--line)}
  .lhash{font-family:var(--mono);font-size:.78rem;color:var(--accent)}
  .lnote{font-size:.88rem;color:var(--muted);margin-top:3px}
  .lstate{font-family:var(--mono);font-size:.7rem;color:var(--good);white-space:nowrap}
  .lstate.todo{color:var(--faint)}
  footer{padding-top:64px;font-size:.82rem;color:var(--faint);
         border-top:1px solid var(--line);margin-top:64px}
"""

EXTRA_CSS = """
  .cover{display:none}
  @media print{
    :root{--paper:#fff;--surface:#fff;--raised:#f4f6f8;--ink:#111;--muted:#444;
          --faint:#666;--line:#ccc;--line-2:#aaa;--accent:#0D6E86;--accent-soft:#e8f2f5;
          --warm:#A8500C;--warm-soft:#f6e9dc;--good:#196B3C;--bad:#9B2C2C;--good-soft:#e6f2ea}
    @page{margin:14mm}
    body{font-size:10.5pt}
    .wrap{max-width:none;padding:0}
    nav.toc{display:none}
    .cover{display:flex;flex-direction:column;gap:6px;padding-bottom:18px;
           border-bottom:2px solid var(--line-2);margin-bottom:8px}
    .cover .f{font-family:var(--mono);font-size:.8rem;color:var(--muted)}
    header.masthead{padding-top:18px;break-after:avoid}
    section{padding-top:26px;break-inside:auto}
    h2,h3{break-after:avoid}
    .mcard,.tcase,.figure,.note,.tablewrap,.models,.driftgrid{break-inside:avoid}
    .stats,ol.steps>li,.nrow,.lrow{break-inside:avoid}
    table{min-width:0}
    .figure svg{min-width:0}
    a{text-decoration:none}
  }
  .tcase{background:var(--surface);border:1px solid var(--line);border-radius:6px;
         padding:18px;display:flex;flex-direction:column;gap:14px;margin-bottom:16px}
  .tchead{display:flex;flex-direction:column;gap:7px}
  .tcsent code{font-size:.92rem;background:transparent;border:0;padding:0;color:var(--ink)}
  .tcgold{font-size:.78rem;color:var(--faint)}
  .tcgold b{color:var(--muted)}
  .tcase .tablewrap{border:0;background:transparent}
  .tcase table{min-width:460px}
  .tcnb{display:grid;grid-template-columns:90px 1fr;gap:10px;align-items:baseline;
        border-top:1px dashed var(--line);padding-top:12px}
  .tcnb .chip{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .tclegend{font-size:.74rem;color:var(--faint);margin-top:-4px}
  .subhead{font-family:var(--sans);font-size:.78rem;letter-spacing:.1em;
           text-transform:uppercase;color:var(--faint);font-weight:600;margin-top:10px}
  .qhead{font-family:var(--mono);font-size:.9rem;color:var(--accent);margin-top:16px}
  .chip b{font-weight:500;color:var(--faint)}
"""
