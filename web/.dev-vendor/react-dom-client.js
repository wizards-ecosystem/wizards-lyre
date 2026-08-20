import { r as reactDomExports } from './chunk-D0qzBqDT.js';

var client = {};

var m = reactDomExports;
if (process.env.NODE_ENV === 'production') {
  client.createRoot = m.createRoot;
  client.hydrateRoot = m.hydrateRoot;
} else {
  var i = m.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;
  client.createRoot = function(c, o) {
    i.usingClientEntryPoint = true;
    try {
      return m.createRoot(c, o);
    } finally {
      i.usingClientEntryPoint = false;
    }
  };
  client.hydrateRoot = function(c, h, o) {
    i.usingClientEntryPoint = true;
    try {
      return m.hydrateRoot(c, h, o);
    } finally {
      i.usingClientEntryPoint = false;
    }
  };
}

const createRoot = client["createRoot"];
const hydrateRoot = client["hydrateRoot"];

export { createRoot, client as default, hydrateRoot };
