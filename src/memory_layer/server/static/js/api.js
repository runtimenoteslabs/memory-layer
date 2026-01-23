/**
 * Memory Layer API Client
 *
 * Provides methods for interacting with the Memory Layer REST API.
 */

class MemoryLayerAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    /**
     * Make an API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle empty responses
            const text = await response.text();
            return text ? JSON.parse(text) : null;
        } catch (error) {
            console.error(`API Error: ${endpoint}`, error);
            throw error;
        }
    }

    // =========================================================================
    // Memory Operations
    // =========================================================================

    /**
     * Get all memories with optional filters
     */
    async getMemories(params = {}) {
        const query = new URLSearchParams();
        if (params.category) query.set('category', params.category);
        if (params.project) query.set('project', params.project);
        if (params.limit) query.set('limit', params.limit);
        if (params.offset) query.set('offset', params.offset);

        const queryStr = query.toString();
        const response = await this.request(`/memories${queryStr ? '?' + queryStr : ''}`);
        // API returns {count, memories} - extract the array
        return response.memories || response;
    }

    /**
     * Get a single memory by ID
     */
    async getMemory(id) {
        return this.request(`/memories/${id}`);
    }

    /**
     * Create a new memory
     */
    async createMemory(data) {
        return this.request('/memories', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * Update a memory
     */
    async updateMemory(id, data) {
        return this.request(`/memories/${id}`, {
            method: 'PATCH',
            body: JSON.stringify(data)
        });
    }

    /**
     * Delete a memory
     */
    async deleteMemory(id) {
        return this.request(`/memories/${id}`, {
            method: 'DELETE'
        });
    }

    // =========================================================================
    // Search
    // =========================================================================

    /**
     * Search memories
     * @param {string} query - Search query
     * @param {Object} options - Search options
     * @param {string} options.searchType - 'semantic' or 'keyword'
     */
    async search(query, options = {}) {
        const body = {
            query,
            limit: options.limit || 20,
            categories: options.categories || null,
            project: options.project || null,
            search_type: options.searchType || 'semantic'
        };
        console.log('API search request:', body);
        const response = await this.request('/memories/search', {
            method: 'POST',
            body: JSON.stringify(body)
        });
        console.log('API search response:', response);
        // API returns {count, results} - extract the array
        return response.results || response;
    }

    // =========================================================================
    // Outcomes
    // =========================================================================

    /**
     * Record an outcome for memories
     */
    async recordOutcome(memoryIds, outcome) {
        return this.request('/memories/outcome', {
            method: 'POST',
            body: JSON.stringify({
                memory_ids: Array.isArray(memoryIds) ? memoryIds : [memoryIds],
                outcome
            })
        });
    }

    // =========================================================================
    // Stats & Health
    // =========================================================================

    /**
     * Get memory statistics
     */
    async getStats(project = null) {
        const query = project ? `?project=${encodeURIComponent(project)}` : '';
        return this.request(`/stats${query}`);
    }

    /**
     * Health check
     */
    async healthCheck() {
        return this.request('/health');
    }

    // =========================================================================
    // Context
    // =========================================================================

    /**
     * Get context
     */
    async getContext(project = null) {
        const query = project ? `?project=${encodeURIComponent(project)}` : '';
        return this.request(`/context${query}`);
    }

    // =========================================================================
    // Beads Integration (Legacy - use unified tasks API instead)
    // =========================================================================

    /**
     * Sync Beads tasks
     */
    async beadsSync() {
        return this.request('/beads/sync', { method: 'POST' });
    }

    /**
     * Get Beads context
     */
    async beadsContext(taskId = null) {
        const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
        return this.request(`/beads/context${query}`);
    }

    /**
     * Get Beads tasks
     */
    async beadsTasks() {
        return this.request('/beads/tasks');
    }

    /**
     * Get Beads stats
     */
    async beadsStats() {
        return this.request('/beads/stats');
    }

    // =========================================================================
    // Unified Tasks Integration (Phase 7 - Claude Code Tasks Adapter)
    // =========================================================================

    /**
     * Get tasks from all sources (Beads and Claude Code)
     * @param {Object} options - Filter options
     * @param {string} options.source - Filter by source: 'beads' or 'claude_code'
     * @param {string} options.status - Filter by status
     * @param {number} options.limit - Max tasks to return
     */
    async getTasks(options = {}) {
        const query = new URLSearchParams();
        if (options.source) query.set('source', options.source);
        if (options.status) query.set('task_status', options.status);
        if (options.limit) query.set('limit', options.limit);

        const queryStr = query.toString();
        return this.request(`/tasks${queryStr ? '?' + queryStr : ''}`);
    }

    /**
     * Get a specific task by ID
     */
    async getTask(taskId) {
        return this.request(`/tasks/${encodeURIComponent(taskId)}`);
    }

    /**
     * Sync task outcomes
     * @param {Object} options - Sync options
     * @param {string} options.source - Sync specific source only
     * @param {string} options.taskId - Sync specific task only
     */
    async syncTasks(options = {}) {
        const body = {};
        if (options.source) body.source = options.source;
        if (options.taskId) body.task_id = options.taskId;

        return this.request('/tasks/sync', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    /**
     * Get unified task context
     * @param {string} taskId - Task ID (optional, uses current if not provided)
     * @param {string} source - Source filter (optional)
     */
    async getTaskContext(taskId = null, source = null) {
        const query = new URLSearchParams();
        if (taskId) query.set('task_id', taskId);
        if (source) query.set('source', source);

        const queryStr = query.toString();
        return this.request(`/tasks/context${queryStr ? '?' + queryStr : ''}`);
    }

    /**
     * Get memories linked to a task
     */
    async getTaskMemories(taskId) {
        return this.request(`/tasks/${encodeURIComponent(taskId)}/memories`);
    }

    /**
     * Get unified task statistics
     */
    async getTasksStats() {
        return this.request('/tasks/stats');
    }

    // =========================================================================
    // Export
    // =========================================================================

    /**
     * Export memories as JSON
     */
    async exportMemories(options = {}) {
        const memories = await this.getMemories({ limit: 100 });
        return memories;
    }
}

// Global API instance
const api = new MemoryLayerAPI();
