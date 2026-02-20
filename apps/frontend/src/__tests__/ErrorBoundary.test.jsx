import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useState } from 'react';
import ErrorBoundary from '../components/ErrorBoundary';

// A child that throws during render
function Bomb({ shouldThrow }) {
  if (shouldThrow) throw new Error('Test render error');
  return <div>Sağlıklı içerik</div>;
}

// Suppress console.error noise from intentional throws
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText('Sağlıklı içerik')).toBeInTheDocument();
  });

  it('shows fallback UI when child throws', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/beklenmeyen bir hata oluştu/i)).toBeInTheDocument();
  });

  it('displays the error message in the pre block', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/test render error/i)).toBeInTheDocument();
  });

  it('renders a "Tekrar Dene" reset button', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByRole('button', { name: /tekrar dene/i })).toBeInTheDocument();
  });

  it('renders a "Sayfayı Yenile" button', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByRole('button', { name: /sayfayı yenile/i })).toBeInTheDocument();
  });

  it('"Tekrar Dene" button is clickable without crashing the test', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/beklenmeyen bir hata oluştu/i)).toBeInTheDocument();
    // Clicking reset does not throw an unhandled error itself
    expect(() => fireEvent.click(screen.getByRole('button', { name: /tekrar dene/i }))).not.toThrow();
  });

  it('recovers fully after reset when new non-throwing children are provided', () => {
    // Use a stateful parent to toggle the child; key-remount trick for ErrorBoundary
    function Toggler() {
      const [shouldThrow, setShouldThrow] = useState(true);
      return (
        <div>
          <button onClick={() => setShouldThrow(false)}>Fix</button>
          <ErrorBoundary key={shouldThrow ? 'err' : 'ok'}>
            <Bomb shouldThrow={shouldThrow} />
          </ErrorBoundary>
        </div>
      );
    }
    render(<Toggler />);
    expect(screen.getByText(/beklenmeyen bir hata oluştu/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /fix/i }));
    expect(screen.getByText('Sağlıklı içerik')).toBeInTheDocument();
  });

  it('has informative subtitle text', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/sayfayı yenileyerek tekrar deneyebilirsiniz/i)).toBeInTheDocument();
  });
});
