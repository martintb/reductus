import 'https://unpkg.com/d3@5.16.0/dist/d3.min.js';
let d3 = window.d3;
export {d3};

import {zip, Inflater, Deflater} from 'https://cdn.jsdelivr.net/gh/bmaranville/zip.js@0.0.3/WebContent/index.js';
zip.Inflater = Inflater;
zip.Deflater = Deflater;
export {zip};

export {default as Tree} from 'https://cdn.jsdelivr.net/gh/bmaranville/treejs@latest/src/index.js';
//export {default as Tree} from './treejs/src/index.js';

import { default as Vue } from 'https://cdn.jsdelivr.net/npm/vue@2.6.11/dist/vue.esm.browser.js';
import { default as VueMaterial } from 'https://cdn.skypack.dev/vue-material@latest';
Vue.use(VueMaterial);
// vuedraggable is available as a single-file esm source, Vue.Draggable/src/vuedraggable.js
// which depends only on Sortable, which can be found at 
// https://github.com/SortableJS/sortablejs/blob/master/modular/sortable.complete.esm.js
export { default as vuedraggable } from 'https://dev.jspm.io/npm:vuedraggable@2.23.2/src/vuedraggable.js';
export { Vue }

import * as messagepack from "https://unpkg.com/messagepack@1.1.11/dist/messagepack.es.js";
//import { default as messagepack } from "https://dev.jspm.io/npm:messagepack@1.1";
export { messagepack };
// PouchDB can be replaced with idb from https://github.com/jakearchibald/idb 
// since we are not using the sync features of pouch
import { default as PouchDB } from "https://dev.jspm.io/npm:pouchdb-browser@7";
export { PouchDB };
export {
    xyChart,
    heatChart,
    heatChartMultiMasked,
    colormap_names,
    get_colormap,
    dataflowEditor,
    extend,
    rectangleSelect,
    xSliceInteractor,
    ySliceInteractor,
    rectangleInteractor,
    ellipseInteractor,
    angleSliceInteractor,
    monotonicFunctionInteractor,
    scaleInteractor,
    rectangleSelectPoints
}  from 'https://cdn.jsdelivr.net/gh/usnistgov/d3-science@0.2.14/src/index.js';
//}  from './d3-science/src/index.js';

export {default as Split} from "https://dev.jspm.io/split.js@1.5.11";
export {default as sha1} from "https://dev.jspm.io/sha1@1.1.1";
export const template_editor_url = "template_editor_live_dev.html";
export {default as json_patch} from "https://cdn.skypack.dev/json-patch-es6";
