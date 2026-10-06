/**
 * Automaton AI - Floating Chatbot Embed Script
 * Embed on any webpage with:
 * <script src="http://127.0.0.1:8000/embed.js"></script>
 */
(function() {
  const SCRIPT_URL = document.currentScript ? document.currentScript.src : "http://127.0.0.1:8000/embed.js";
  const BASE_URL = SCRIPT_URL.substring(0, SCRIPT_URL.lastIndexOf('/'));

  const iframe = document.createElement('iframe');
  iframe.src = `${BASE_URL}/index.html`;
  iframe.title = "Automaton AI Assistant";
  iframe.style.position = 'fixed';
  iframe.style.bottom = '16px';
  iframe.style.right = '16px';
  iframe.style.width = '440px';
  iframe.style.height = '680px';
  iframe.style.border = 'none';
  iframe.style.zIndex = '2147483647';
  iframe.style.backgroundColor = 'transparent';
  iframe.style.colorScheme = 'normal';
  iframe.allow = 'clipboard-write';

  document.body.appendChild(iframe);
})();
