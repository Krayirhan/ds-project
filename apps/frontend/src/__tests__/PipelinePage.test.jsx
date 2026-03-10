import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import PipelinePage from '../components/PipelinePage';

describe('PipelinePage', () => {
  it('renders without crashing', () => {
    const { container } = render(<PipelinePage />);
    expect(container).toBeTruthy();
  });

  it('renders pipeline documentation content', () => {
    render(<PipelinePage />);
    // PipelinePage is a static documentation page — should render text
    expect(document.body.textContent.length).toBeGreaterThan(0);
  });

  it('renders pipeline stage information', () => {
    const { container } = render(<PipelinePage />);
    // Should contain pipeline-related headings or sections
    const headings = container.querySelectorAll('h1, h2, h3, h4');
    expect(headings.length).toBeGreaterThan(0);
  });
});
