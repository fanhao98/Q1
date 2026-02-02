const fs = require('fs');

function extractFunction(source, name) {
  const idx = source.indexOf(`function ${name}(`);
  if (idx < 0) throw new Error(`missing function ${name}`);
  let i = idx;
  while (i < source.length && source[i] !== '{') i++;
  if (i >= source.length) throw new Error(`bad function ${name}`);
  let depth = 0;
  let end = i;
  for (; end < source.length; end++) {
    const ch = source[end];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        end++;
        break;
      }
    }
  }
  return source.slice(idx, end);
}

const html = fs.readFileSync('quant_frontend.html', 'utf8');
const calcSrc = extractFunction(html, 'calculateBacktest');
const updateSrc = extractFunction(html, 'updateOptimizationSummaryTableV2');

const sandbox = {};
sandbox.global = sandbox;
sandbox.window = sandbox;
sandbox.document = {
  _el: { innerHTML: '' },
  getElementById() {
    return this._el;
  }
};

const prelude = `
${calcSrc}
${updateSrc}
`;

const fn = new Function('global', 'window', 'document', `${prelude}; return { calculateBacktest, updateOptimizationSummaryTableV2 };`);
const { calculateBacktest, updateOptimizationSummaryTableV2 } = fn(sandbox, sandbox, sandbox.document);

const data = Array.from({ length: 200 }, (_, i) => ({ close: 10 + i * 0.01 }));
const r1 = calculateBacktest(data, [], Number.NaN);
if (!Number.isFinite(r1.totalReturn)) throw new Error('totalReturn should be finite');
if (!Number.isFinite(r1.initCapital) || r1.initCapital <= 0) throw new Error('initCapital should be defaulted');

const payload = {
  results: {
    foo: {
      name: 'Foo',
      bestParams: { takeProfit: 0.1, stopLoss: 0.05, volumeMulti: 1.2, dynamicTpCallback: 0.03 },
      trainSummary: { totalReturn: { median: null } },
      testSummary: { totalReturn: { median: null } },
      samples: 10
    }
  }
};

updateOptimizationSummaryTableV2(payload);
if (!sandbox.document._el.innerHTML.includes('--')) throw new Error('should render -- when return is missing');

process.stdout.write('ok\n');

